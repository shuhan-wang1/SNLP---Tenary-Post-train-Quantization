"""
PT2-LLM: Main Quantization Entry Point (Optimized)

Analysis based on Ablation Study (Table 2b):
Since SSR (Weight-based) > Hessian-based Reordering, we can aggressively 
optimize the algorithm by relying on Weight Structure for speed.

Modes:
- fast: Hessian-Free. Uses SSR + ITF. (Theory: SSR captures structure better than Hessian)
- balanced: Low-Cost Hessian. 64 samples, Window=256. (Theory: Table 2c & Diagonal Dominance)
- accurate: Full Hessian. 128 samples, Window=512.

Usage:
    python main.py --model Qwen/Qwen3-0.5B --output ./quantized_model --mode fast
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
from quantizer import AsymmetricTernaryQuantizer
from reorder import SSRReorderer, select_next_block_ssr, get_ssr_permutation
from utils import (
    set_seed,
    get_calibration_data,
    save_quantized_model
)

class StopForwardException(Exception):
    pass

class PT2LLMQuantizer:
    def __init__(self,
                 model: nn.Module,
                 tokenizer,
                 model_type: str = 'llama',
                 mode: str = 'accurate',  # fast, balanced, accurate
                 block_size: int = 128,
                 device: str = 'cuda'):
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.block_size = block_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mode = mode

        # --- Optimization Presets based on Paper Ablations ---
        if mode == 'fast':
            # Theory: Table 2b shows SSR (Weight) > Hessian Reorder.
            # We trust Weight Structure completely. No Data needed.
            self.use_hessian = False
            self.num_samples = 0
            self.atq_iter = 2    # Converges fast (Fig 3)
            self.window_size = 0 # Not used
            print(f"🚀 Mode: FAST (Hessian-Free). Relying on SSR structural superiority.")
            
        elif mode == 'balanced':
            # Theory: Table 2c shows 64 samples is almost same as 128.
            # SSR makes Hessian diagonal-dominant, so small window (256) is enough.
            self.use_hessian = True
            self.num_samples = 64
            self.atq_iter = 3
            self.window_size = 256
            print(f"⚖️  Mode: BALANCED. Reduced samples (64) and window (256).")
            
        else: # accurate
            self.use_hessian = True
            self.num_samples = 128
            self.atq_iter = 5
            self.window_size = 512 # Reduced from 2048 as SSR handles long-range dependencies
            print(f"🎯 Mode: ACCURATE. Standard configuration.")

        # Always use SSR (It's the core strength)
        self.use_ssr = True
        self.use_precomputed_ssr = True # K-Means is much faster O(mk) vs O(m^2)
        self.percdamp = 0.01
        self.seed = 42
        set_seed(self.seed)
        
        self.quantized_params: Dict[str, Dict[str, torch.Tensor]] = {}
        
    def get_calibration_data(self) -> List[torch.Tensor]:
        if self.num_samples == 0:
            return []
        print(f"Loading {self.num_samples} calibration samples...")
        return get_calibration_data(
            self.tokenizer,
            dataset_name='wikitext',
            dataset_config='wikitext-2-raw-v1',
            num_samples=self.num_samples, # Use optimized number
            seq_len=2048,
            seed=self.seed
        )
    
    @torch.no_grad()
    def quantize_layer(self,
                       layer: nn.Linear,
                       layer_name: str,
                       calibration_activations: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        W = layer.weight.data.clone().to(self.device)
        n, m = W.shape
        H, H_inv, S_block = None, None, None

        # --- 1. Hessian Calculation (The Bottleneck) ---
        # Skipped entirely in 'fast' mode
        if self.use_hessian and calibration_activations is not None:
            X = calibration_activations
            if X.dim() == 3: X = X.reshape(-1, X.shape[-1])
            
            # O(N*d^2) matrix multiplication
            H = X.T @ X
            H = H / X.shape[0]
            
            damp = self.percdamp * torch.diag(H).mean()
            H.diagonal().add_(damp)
            
            # Inversion for GPTQ
            try:
                H_inv = torch.linalg.cholesky_inverse(torch.linalg.cholesky(H.float()))
            except RuntimeError:
                H_inv = torch.linalg.pinv(H.float())
        
        # --- 2. SSR Reordering (The Hero) ---
        # We always use this because it's better than Hessian-based reordering (Table 2b)
        # Using Precomputed K-Means (Fast)
        perm = get_ssr_permutation(W, self.block_size)
        W = W[:, perm]
        
        if H is not None:
            H = H[perm][:, perm]
            H_inv = H_inv[perm][:, perm]

        # --- 3. Block-wise Quantization ---
        atq = AsymmetricTernaryQuantizer(max_iter=self.atq_iter)
        
        T_full = torch.zeros(n, m, dtype=torch.int8, device=self.device)
        alpha_list = []
        mu_list = []
        
        for i in range(0, m, self.block_size):
            end = min(i + self.block_size, m)
            block_cols = torch.arange(i, end, device=self.device)
            
            W_block = W[:, block_cols]
            
            # Prepare S for AGA (Only in balanced/accurate mode)
            if H is not None:
                S_block = H[block_cols][:, block_cols]
            
            # ATQ Quantization (ITF + Optional AGA)
            # In 'fast' mode, S_block is None, so it runs Pure ITF (Weight Only)
            alpha_b, mu_b, T_b = atq.quantize(W_block, X=None, S=S_block)
            
            alpha_list.append(alpha_b)
            mu_list.append(mu_b)
            T_full[:, block_cols] = T_b.to(torch.int8)
            
            # --- 4. GPTQ Error Compensation ---
            # Skipped in 'fast' mode
            if H_inv is not None:
                W_quant_block = alpha_b * T_b + mu_b
                error = W_block - W_quant_block
                
                remaining_cols = torch.arange(end, m, device=self.device)
                if len(remaining_cols) > 0:
                    # Optimized Window Size
                    limit = min(len(remaining_cols), self.window_size)
                    update_cols = remaining_cols[:limit]
                    
                    H_inv_block = H_inv[block_cols][:, update_cols]
                    H_inv_diag = H_inv[block_cols, block_cols].unsqueeze(1)
                    
                    # Update weights
                    W[:, update_cols] -= error @ (H_inv_block / H_inv_diag)

        alpha = torch.cat(alpha_list, dim=1)
        mu = torch.cat(mu_list, dim=1)

        return {'alpha': alpha.cpu(), 'mu': mu.cpu(), 'T': T_full.cpu(), 'perm': perm.cpu()}

    def quantize(self):
        print("="*60)
        print(f"PT2-LLM Optimization | Mode: {self.mode.upper()}")
        print("="*60)
        
        # Phase 1: Input Caching (Only if needed)
        inps = []
        cache = {'attention_mask': None, 'position_ids': None}
        
        if self.use_hessian:
            print("Phase 1: Input Caching...")
            calib_data = self.get_calibration_data()
            layers = get_llm_layers(self.model, self.model_type)
            
            # [FIXED] Updated Catcher to proxy attributes
            class Catcher(nn.Module):
                def __init__(self, module):
                    super().__init__()
                    self.module = module
                def forward(self, *args, **kwargs):
                    if len(args) > 0: inps.append(args[0].detach())
                    elif 'hidden_states' in kwargs: inps.append(kwargs['hidden_states'].detach())
                    if cache['attention_mask'] is None: 
                         cache['attention_mask'] = kwargs.get('attention_mask')
                         cache['position_ids'] = kwargs.get('position_ids')
                    raise StopForwardException
                
                # Critical Fix: Forward attribute access to the wrapped module
                # This prevents crashes when model checks for 'attention_type' etc.
                def __getattr__(self, name):
                    try:
                        return super().__getattr__(name)
                    except AttributeError:
                        return getattr(self.module, name)
            
            layers[0] = Catcher(layers[0])
            try:
                with torch.no_grad():
                    for batch in calib_data:
                        self.model(batch.to(self.device))
            except StopForwardException:
                pass
            layers[0] = layers[0].module # Restore
        else:
            print("Phase 1: Skipped (Hessian-Free)")

        # Phase 2: Quantization
        layers = get_llm_layers(self.model, self.model_type)
        print(f"Phase 2: Quantizing {len(layers)} layers...")
        
        for i, layer in tqdm(enumerate(layers), total=len(layers)):
            layer = layer.to(self.device)
            linear_layers = find_linear_layers(layer)
            
            # Forward pass to get inputs for this layer (if using Hessian)
            layer_inputs = {name: [] for name in linear_layers}
            if self.use_hessian and len(inps) > 0:
                def get_hook(name):
                    def hook(m, x, y): layer_inputs[name].append(x[0].detach())
                    return hook
                handles = [l.register_forward_hook(get_hook(n)) for n, l in linear_layers.items()]
                
                # Run inputs through layer
                new_inps = []
                with torch.no_grad():
                    for batch_idx, inp in enumerate(inps):
                        out = layer(inp.to(self.device), **{k:v for k,v in cache.items() if v is not None})
                        if isinstance(out, tuple): out = out[0]
                        new_inps.append(out)
                inps = new_inps # Pass to next layer
                for h in handles: h.remove()
            
            # Quantize Sub-layers
            for name, linear in linear_layers.items():
                full_name = f"layer_{i}.{name}"
                X = torch.cat(layer_inputs[name], dim=0) if layer_inputs[name] else None
                
                # Core Quantization Call
                params = self.quantize_layer(linear, full_name, X)
                self.quantized_params[full_name] = params
                
                # Update weights (Fake Quantization)
                T = params['T'].float().to(self.device)
                alpha = params['alpha'].to(self.device)
                mu = params['mu'].to(self.device)
                perm = params['perm'].to(self.device)
                
                # Dequantize Logic inline
                W_new = torch.zeros_like(linear.weight.data)
                for b in range(alpha.shape[1]):
                    s, e = b*self.block_size, min((b+1)*self.block_size, W_new.shape[1])
                    W_new[:, s:e] = alpha[:, b:b+1] * T[:, s:e] + mu[:, b:b+1]
                
                # Restore order
                inv_perm = torch.argsort(perm)
                linear.weight.data = W_new[:, inv_perm]
                
                del X
            
            gc.collect()
            torch.cuda.empty_cache()

        return self.quantized_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output', type=str, default='./quantized_model')
    parser.add_argument('--mode', type=str, default='accurate', choices=['fast', 'balanced', 'accurate'], 
                        help="Optimization mode based on ablation analysis")
    parser.add_argument('--block_size', type=int, default=128)
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    print(f"Loading model: {args.model}")
    model, tokenizer = load_model_for_quantization(args.model, device=args.device)
    
    quantizer = PT2LLMQuantizer(
        model=model,
        tokenizer=tokenizer,
        mode=args.mode,
        block_size=args.block_size,
        device=args.device
    )
    
    quantizer.quantize()
    save_quantized_model(model, f"{args.output}/model.pt", quantizer.quantized_params)
    tokenizer.save_pretrained(args.output)

if __name__ == '__main__':
    main()