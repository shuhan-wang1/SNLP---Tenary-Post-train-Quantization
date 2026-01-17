"""
PT2-LLM: Main Quantization Entry Point

Post-Training Ternarization for Large Language Models

Usage:
    python main.py --model meta-llama/Llama-2-7b-hf --output ./quantized_model

Reference: PT2-LLM: Post-Training Ternarization for Large Language Models
"""

import argparse
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import gc
import time

from model import (
    load_model_for_quantization, 
    get_llm_layers, 
    get_model_type,
    find_linear_layers,
    TernaryLinear,
    compute_model_size
)
from quantizer import AsymmetricTernaryQuantizer, compute_quantization_error, compute_output_error
from reorder import SSRReorderer, select_next_block_ssr
from gptq import GPTQ
from utils import (
    set_seed,
    get_calibration_data,
    evaluate_perplexity,
    compute_bits_per_weight,
    save_quantized_model
)


class PT2LLMQuantizer:
    """
    PT2-LLM: Post-Training Ternarization Quantizer
    
    Combines:
    1. Asymmetric Ternary Quantizer (ATQ) with ITF and AGA
    2. Structural Similarity-based Reordering (SSR)
    3. GPTQ-style block-wise quantization with error compensation
    """
    
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 model_type: str = 'llama',
                 block_size: int = 128,
                 num_calibration_samples: int = 128,
                 seq_len: int = 2048,
                 use_ssr: bool = True,
                 percdamp: float = 0.01,
                 seed: int = 42,
                 device: str = 'cuda'):
        """
        Args:
            model: Model to quantize
            tokenizer: Associated tokenizer
            model_type: Type of model architecture
            block_size: Columns per quantization block
            num_calibration_samples: Number of calibration samples
            seq_len: Sequence length for calibration
            use_ssr: Whether to use SSR reordering
            percdamp: GPTQ dampening factor
            seed: Random seed
            device: Device for computation
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.block_size = block_size
        self.num_calibration_samples = num_calibration_samples
        self.seq_len = seq_len
        self.use_ssr = use_ssr
        self.percdamp = percdamp
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        set_seed(seed)
        
        # Store quantization results
        self.quantized_params: Dict[str, Dict[str, torch.Tensor]] = {}
        
    def get_calibration_data(self) -> List[torch.Tensor]:
        """Load calibration data."""
        print(f"Loading {self.num_calibration_samples} calibration samples...")
        return get_calibration_data(
            self.tokenizer,
            dataset_name='wikitext',
            dataset_config='wikitext-2-raw-v1',
            num_samples=self.num_calibration_samples,
            seq_len=self.seq_len,
            seed=self.seed
        )
    
    @torch.no_grad()
    def quantize_layer(self, 
                       layer: nn.Linear,
                       layer_name: str,
                       calibration_activations: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Quantize a single linear layer using PT2-LLM method.
        
        Args:
            layer: Linear layer to quantize
            layer_name: Name of the layer
            calibration_activations: Input activations for calibration
            
        Returns:
            Dictionary containing quantized parameters
        """
        W = layer.weight.data.clone().to(self.device)
        n, m = W.shape  # (out_features, in_features)
        
        # Reshape activations: (B, L, m) -> (B*L, m)
        if calibration_activations.dim() == 3:
            X = calibration_activations.reshape(-1, calibration_activations.shape[-1]).to(self.device)
        else:
            X = calibration_activations.to(self.device)
        
        # Compute Hessian: H = X^T X
        H = X.T @ X
        H = H / X.shape[0]
        
        # Add dampening
        damp = self.percdamp * torch.diag(H).mean()
        H.diagonal().add_(damp)
        
        # Compute Hessian inverse
        H = H.float()
        try:
            H_chol = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_chol)
        except RuntimeError:
            H_inv = torch.linalg.pinv(H)
        print(H.device, H_inv.device)  # 应该是 cuda
        # Initialize storage
        T_full = torch.zeros(n, m, dtype=torch.int8, device=self.device)
        alpha_list = []
        mu_list = []
        
        # Initialize column order
        if self.use_ssr:
            remaining_indices = torch.arange(m, device=self.device)
            perm_list = []
        else:
            perm = torch.arange(m, device=self.device)
        
        # ATQ quantizer
        atq = AsymmetricTernaryQuantizer()
        
        # Block-wise quantization
        processed_cols = 0
        while processed_cols < m:
            # Select block columns
            if self.use_ssr and len(remaining_indices) > 0:
                block_indices, remaining_indices = select_next_block_ssr(
                    W, remaining_indices, self.block_size
                )
                perm_list.extend(block_indices.tolist())
            else:
                end_idx = min(processed_cols + self.block_size, m)
                block_indices = torch.arange(processed_cols, end_idx, device=self.device)
            
            block_size = len(block_indices)
            if block_size == 0:
                break
            
            # Extract block
            W_block = W[:, block_indices]
            X_block = X[:, block_indices]
            
            # Apply ATQ (ITF + AGA)
            alpha_b, mu_b, T_b = atq.quantize(W_block, X_block)
            
            # Store results
            alpha_list.append(alpha_b)
            mu_list.append(mu_b)
            T_full[:, block_indices] = T_b.to(torch.int8)
            
            # GPTQ error compensation
            W_quant_block = alpha_b * T_b + mu_b
            quant_error = W_block - W_quant_block
            
            # Propagate error to remaining columns
            if self.use_ssr:
                remaining_cols = remaining_indices
            else:
                remaining_cols = torch.arange(processed_cols + block_size, m, device=self.device)
            
            # ✅ 优化后的实现：直接使用矩阵乘法一次性更新所有剩余列
            if len(remaining_cols) > 0:
                # 1. 提取当前块对应的 H_inv 行：H_inv[block, remaining]
                # shape: (block_size, num_remaining)
                H_inv_block_rem = H_inv[block_indices][:, remaining_cols]
                
                # 2. 提取对角线元素并调整维度
                # shape: (block_size, 1)
                H_inv_diag = H_inv[block_indices, block_indices].unsqueeze(1).clamp(min=1e-8)
                
                # 3. 计算系数矩阵 (H_inv_ij / H_inv_ii)
                # shape: (block_size, num_remaining)
                coefficients = H_inv_block_rem / H_inv_diag
                
                # 4. 向量化更新：W_rem -= Error_block @ Coefficients
                # quant_error shape: (rows, block_size)
                # update shape: (rows, num_remaining)
                W[:, remaining_cols] -= quant_error @ coefficients
            processed_cols += block_size
        
        # Finalize permutation
        if self.use_ssr:
            perm = torch.tensor(perm_list, device=self.device, dtype=torch.long)
        
        # Stack block parameters
        alpha = torch.cat(alpha_list, dim=1)
        mu = torch.cat(mu_list, dim=1)
        
        return {
            'alpha': alpha.cpu(),
            'mu': mu.cpu(),
            'T': T_full.cpu(),
            'perm': perm.cpu()
        }
    
    def quantize(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Quantize the entire model layer by layer.
        
        Returns:
            Dictionary mapping layer names to quantized parameters
        """
        print("="*60)
        print("PT2-LLM: Post-Training Ternarization")
        print("="*60)
        
        start_time = time.time()
        
        # Get calibration data
        calibration_samples = self.get_calibration_data()
        
        # Get transformer layers
        layers = get_llm_layers(self.model, self.model_type)
        num_layers = len(layers)
        
        print(f"\nQuantizing {num_layers} transformer layers...")
        print(f"Block size: {self.block_size}, SSR: {self.use_ssr}")
        print("-"*60)
        
        # Process layer by layer
        for layer_idx, layer in enumerate(tqdm(layers, desc="Layers")):
            layer_activations = {}
            hooks = []
            
            # Register hooks to capture activations
            def make_hook(name):
                def hook(module, inp, out):
                    if isinstance(inp, tuple):
                        inp = inp[0]
                    if name not in layer_activations:
                        layer_activations[name] = []
                    layer_activations[name].append(inp.detach())
                return hook
            
            # Find linear layers in this transformer layer
            linear_layers = find_linear_layers(layer)
            
            for name, linear in linear_layers.items():
                hooks.append(linear.register_forward_hook(make_hook(name)))
            
            # Collect activations through forward pass
            self.model.eval()
            with torch.no_grad():
                for sample in calibration_samples:
                    sample = sample.to(self.device)
                    self.model(sample)
            
            # Remove hooks
            for hook in hooks:
                hook.remove()
            
            # Quantize each linear layer
            for name, linear in linear_layers.items():
                full_name = f"layer_{layer_idx}.{name}"
                
                if name in layer_activations and len(layer_activations[name]) > 0:
                    activations = torch.cat(layer_activations[name], dim=0)
                    params = self.quantize_layer(linear, full_name, activations)
                    self.quantized_params[full_name] = params
                    
                    # Replace weight with quantized version
                    W_quant = self._dequantize_weight(params)
                    linear.weight.data = W_quant.to(linear.weight.device, linear.weight.dtype)
            
            # Clear cache
            del layer_activations
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        elapsed = time.time() - start_time
        print("-"*60)
        print(f"Quantization completed in {elapsed:.1f}s")
        print(f"Total quantized layers: {len(self.quantized_params)}")
        
        return self.quantized_params
    
    def _dequantize_weight(self, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Dequantize weights from stored parameters."""
        T = params['T'].float()
        alpha = params['alpha']
        mu = params['mu']
        perm = params['perm']
        
        n, m = T.shape
        W = torch.zeros(n, m, dtype=alpha.dtype)
        
        num_blocks = alpha.shape[1]
        for b in range(num_blocks):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, m)
            
            T_block = T[:, start:end]
            W[:, start:end] = alpha[:, b:b+1] * T_block + mu[:, b:b+1]
        
        # Apply inverse permutation
        inv_perm = torch.argsort(perm)
        W = W[:, inv_perm]
        
        return W


def main():
    parser = argparse.ArgumentParser(description='PT2-LLM: Post-Training Ternarization for LLMs')
    
    # Model arguments
    parser.add_argument('--model', type=str, required=True,
                        help='HuggingFace model name or path')
    parser.add_argument('--output', type=str, default='./quantized_model',
                        help='Output directory for quantized model')
    
    # Quantization arguments
    parser.add_argument('--block_size', type=int, default=128,
                        help='Block size for quantization (default: 128)')
    parser.add_argument('--num_samples', type=int, default=128,
                        help='Number of calibration samples (default: 128)')
    parser.add_argument('--seq_len', type=int, default=2048,
                        help='Sequence length for calibration (default: 2048)')
    parser.add_argument('--no_ssr', action='store_true',
                        help='Disable SSR reordering')
    parser.add_argument('--percdamp', type=float, default=0.01,
                        help='GPTQ dampening factor (default: 0.01)')
    
    # Evaluation arguments
    parser.add_argument('--eval', action='store_true',
                        help='Evaluate perplexity after quantization')
    parser.add_argument('--eval_dataset', type=str, default='wikitext',
                        help='Dataset for evaluation (default: wikitext)')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    
    args = parser.parse_args()
    
    # Load model
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_for_quantization(
        args.model,
        device=args.device,
        dtype=torch.float16
    )
    
    model_type = get_model_type(args.model)
    print(f"Model type: {model_type}")
    
    # Compute original size
    orig_size = compute_model_size(model)
    print(f"Original model size: {orig_size:.2f} GB")
    
    # Create quantizer
    quantizer = PT2LLMQuantizer(
        model=model,
        tokenizer=tokenizer,
        model_type=model_type,
        block_size=args.block_size,
        num_calibration_samples=args.num_samples,
        seq_len=args.seq_len,
        use_ssr=not args.no_ssr,
        percdamp=args.percdamp,
        seed=args.seed,
        device=args.device
    )
    
    # Quantize
    quantized_params = quantizer.quantize()
    
    # Compute quantized size
    quant_size = compute_model_size(model)
    bits_per_weight = compute_bits_per_weight(model)
    
    print(f"\nQuantized model size: {quant_size:.2f} GB")
    print(f"Compression ratio: {orig_size/quant_size:.2f}x")
    print(f"Average bits per weight: {bits_per_weight:.2f}")
    
    # Evaluate if requested
    if args.eval:
        print(f"\nEvaluating perplexity on {args.eval_dataset}...")
        ppl = evaluate_perplexity(
            model, tokenizer,
            dataset_name=args.eval_dataset,
            seq_len=args.seq_len,
            device=torch.device(args.device)
        )
        print(f"Perplexity: {ppl:.2f}")
    
    # Save model
    print(f"\nSaving quantized model to {args.output}")
    save_quantized_model(model, f"{args.output}/model.pt", quantized_params)
    tokenizer.save_pretrained(args.output)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
