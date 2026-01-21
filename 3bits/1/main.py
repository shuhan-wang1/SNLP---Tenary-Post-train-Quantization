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
from quantizer import AsymmetricTernaryQuantizer, compute_quantization_error, compute_output_error, compute_siph_hessian
from reorder import SSRReorderer, select_next_block_ssr, get_ssr_permutation
from gptq import GPTQ, WINDOW_SIZE
from utils import (
    set_seed,
    get_calibration_data,
    evaluate_perplexity,
    compute_bits_per_weight,
    save_quantized_model,
    compute_compressed_size,
    compute_packed_size
)


class StopForwardException(Exception):
    """Exception used to stop forward pass after capturing first layer inputs."""
    pass


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
                 use_precomputed_ssr: bool = True,
                 window_size: int = WINDOW_SIZE,
                 percdamp: float = 0.01,
                 seed: int = 42,
                 device: str = 'cuda',
                 mode: str = 'default',
                 siph_gamma: float = 0.5):
        """
        Args:
            model: Model to quantize
            tokenizer: Associated tokenizer
            model_type: Type of model architecture
            block_size: Columns per quantization block
            num_calibration_samples: Number of calibration samples
            seq_len: Sequence length for calibration
            use_ssr: Whether to use SSR reordering
            use_precomputed_ssr: Use K-Means precomputed SSR (faster) vs dynamic SSR
            window_size: Window size for GPTQ error propagation (default 2048)
            percdamp: GPTQ dampening factor
            seed: Random seed
            device: Device for computation
            mode: Quantization mode ('default' or 'siph')
                  - 'default': Standard PT2-LLM with calibration data
                  - 'siph': Data-free SIPH (SSR-Induced Pseudo-Hessian)
            siph_gamma: Exponent for SIPH uniqueness term (default 0.5)
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.block_size = block_size
        self.num_calibration_samples = num_calibration_samples
        self.seq_len = seq_len
        self.use_ssr = use_ssr
        self.use_precomputed_ssr = use_precomputed_ssr
        self.window_size = window_size
        self.percdamp = percdamp
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.mode = mode
        self.siph_gamma = siph_gamma

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

        Optimizations applied:
        1. K-Means precomputed SSR (O(m*k) instead of O(m^2))
        2. Windowed error propagation (only update nearby columns)
        3. Reduced ITF iterations (5 instead of 100)

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

        # Initialize storage
        T_full = torch.zeros(n, m, dtype=torch.int8, device=self.device)
        alpha_list = []
        mu_list = []

        # ========== SSR Reordering Strategy ==========
        if self.use_ssr:
            if self.use_precomputed_ssr:
                # Optimized: Precompute full permutation using K-Means clustering
                perm = get_ssr_permutation(W, self.block_size)
                # Reorder W, X, H, H_inv according to permutation
                W = W[:, perm]
                X = X[:, perm]
                H = H[perm][:, perm]
                H_inv = H_inv[perm][:, perm]
            else:
                # Original: Dynamic block selection (slower)
                remaining_indices = torch.arange(m, device=self.device)
                perm_list = []
        else:
            perm = torch.arange(m, device=self.device)

        # ATQ quantizer (with reduced iterations: default now 5 instead of 100)
        atq = AsymmetricTernaryQuantizer()

        # Block-wise quantization
        processed_cols = 0
        while processed_cols < m:
            # Select block columns
            if self.use_ssr and not self.use_precomputed_ssr:
                # Dynamic SSR
                block_indices, remaining_indices = select_next_block_ssr(
                    W, remaining_indices, self.block_size
                )
                perm_list.extend(block_indices.tolist())
                block_size = len(block_indices)
            else:
                # Sequential (either no SSR or precomputed SSR already reordered W)
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

            # ========== Windowed Error Propagation ==========
            if self.use_ssr and not self.use_precomputed_ssr:
                remaining_cols = remaining_indices
            else:
                remaining_cols = torch.arange(processed_cols + block_size, m, device=self.device)

            if len(remaining_cols) > 0:
                # Windowed update: only update the nearest window_size columns
                # Exploits diagonal-dominant structure of H_inv after SSR reordering
                limit = min(len(remaining_cols), self.window_size)
                update_cols = remaining_cols[:limit]

                # Extract H_inv[block, update_cols]
                H_inv_block_rem = H_inv[block_indices][:, update_cols]

                # Extract diagonal elements
                H_inv_diag = H_inv[block_indices, block_indices].unsqueeze(1).clamp(min=1e-8)

                # Compute coefficients
                coefficients = H_inv_block_rem / H_inv_diag

                # Vectorized update
                W[:, update_cols] -= quant_error @ coefficients

            processed_cols += block_size

        # Finalize permutation
        if self.use_ssr and not self.use_precomputed_ssr:
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

    @torch.no_grad()
    def quantize_layer_siph(self,
                            layer: nn.Linear,
                            layer_name: str) -> Dict[str, torch.Tensor]:
        """
        Quantize a single linear layer using SIPH (Data-Free) mode.

        SIPH (SSR-Induced Pseudo-Hessian) synthesizes a Pseudo-Hessian matrix purely
        from weight statistics, enabling GPTQ-style error compensation without
        any calibration data.

        Args:
            layer: Linear layer to quantize
            layer_name: Name of the layer

        Returns:
            Dictionary containing quantized parameters
        """
        W = layer.weight.data.clone().to(self.device).float()
        n, m = W.shape  # (out_features, in_features)

        # Compute SIPH Pseudo-Hessian (data-free)
        H = compute_siph_hessian(W, self.block_size, gamma=self.siph_gamma)

        # Add dampening for numerical stability
        damp = self.percdamp * torch.diag(H).mean()
        H.diagonal().add_(damp)

        # Compute Hessian inverse using Cholesky decomposition
        try:
            H_chol = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(H_chol)
        except RuntimeError:
            # Fallback to pseudo-inverse if Cholesky fails
            H_inv = torch.linalg.pinv(H)

        # Initialize storage
        T_full = torch.zeros(n, m, dtype=torch.int8, device=self.device)
        alpha_list = []
        mu_list = []

        # ========== SSR Reordering Strategy ==========
        if self.use_ssr:
            if self.use_precomputed_ssr:
                # Optimized: Precompute full permutation using K-Means clustering
                perm = get_ssr_permutation(W, self.block_size)
                # Reorder W, H, H_inv according to permutation
                W = W[:, perm]
                H = H[perm][:, perm]
                H_inv = H_inv[perm][:, perm]
            else:
                # Original: Dynamic block selection (slower)
                remaining_indices = torch.arange(m, device=self.device)
                perm_list = []
        else:
            perm = torch.arange(m, device=self.device)

        # ATQ quantizer
        atq = AsymmetricTernaryQuantizer()

        # Block-wise quantization with GPTQ error compensation
        processed_cols = 0
        while processed_cols < m:
            # Select block columns
            if self.use_ssr and not self.use_precomputed_ssr:
                # Dynamic SSR
                block_indices, remaining_indices = select_next_block_ssr(
                    W, remaining_indices, self.block_size
                )
                perm_list.extend(block_indices.tolist())
                block_size = len(block_indices)
            else:
                # Sequential (either no SSR or precomputed SSR already reordered W)
                end_idx = min(processed_cols + self.block_size, m)
                block_indices = torch.arange(processed_cols, end_idx, device=self.device)
                block_size = len(block_indices)

            if block_size == 0:
                break

            # Extract block
            W_block = W[:, block_indices]

            # Extract Pseudo-Hessian sub-block for AGA
            S_block = H[:, block_indices][block_indices, :]

            # Apply ATQ (ITF + AGA with Pseudo-Hessian)
            # Pass S_block directly as the covariance matrix
            alpha_b, mu_b, T_b = atq.quantize(W_block, X=None, S=S_block)

            # Store results
            alpha_list.append(alpha_b)
            mu_list.append(mu_b)
            T_full[:, block_indices] = T_b.to(torch.int8)

            # GPTQ error compensation
            W_quant_block = alpha_b * T_b + mu_b
            quant_error = W_block - W_quant_block

            # ========== Windowed Error Propagation ==========
            if self.use_ssr and not self.use_precomputed_ssr:
                remaining_cols = remaining_indices
            else:
                remaining_cols = torch.arange(processed_cols + block_size, m, device=self.device)

            if len(remaining_cols) > 0:
                # Windowed update: only update the nearest window_size columns
                limit = min(len(remaining_cols), self.window_size)
                update_cols = remaining_cols[:limit]

                # Extract H_inv[block, update_cols]
                H_inv_block_rem = H_inv[block_indices][:, update_cols]

                # Extract diagonal elements
                H_inv_diag = H_inv[block_indices, block_indices].unsqueeze(1).clamp(min=1e-8)

                # Compute coefficients
                coefficients = H_inv_block_rem / H_inv_diag

                # Vectorized update
                W[:, update_cols] -= quant_error @ coefficients

            processed_cols += block_size

        # Finalize permutation
        if self.use_ssr and not self.use_precomputed_ssr:
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

        Mode-dependent behavior:
        - 'default': Uses calibration data with Hook-based Input Caching
        - 'siph': Data-free quantization using SSR-Induced Pseudo-Hessian

        Returns:
            Dictionary mapping layer names to quantized parameters
        """
        if self.mode == 'siph':
            return self._quantize_siph()
        else:
            return self._quantize_default()

    def _quantize_siph(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Quantize using SIPH (Data-Free) mode.

        No calibration data is needed. The Pseudo-Hessian is computed purely
        from weight statistics.

        Returns:
            Dictionary mapping layer names to quantized parameters
        """
        print("="*60)
        print("PT2-LLM: Post-Training Ternarization (SIPH Data-Free Mode)")
        print("="*60)

        start_time = time.time()

        layers = get_llm_layers(self.model, self.model_type)
        num_layers = len(layers)

        print(f"\nQuantizing {num_layers} transformer layers (Data-Free)...")
        print(f"Block size: {self.block_size}, SSR: {self.use_ssr}, SIPH gamma: {self.siph_gamma}")
        print("-"*60)

        for layer_idx in tqdm(range(num_layers), desc="Quantizing Layers (SIPH)"):
            # Move current layer to compute device
            layer = layers[layer_idx].to(self.device)

            # Find linear layers
            linear_layers = find_linear_layers(layer)

            # Quantize each linear layer using SIPH
            for name, linear in linear_layers.items():
                full_name = f"layer_{layer_idx}.{name}"

                # Run SIPH quantization (no calibration data needed)
                params = self.quantize_layer_siph(linear, full_name)
                self.quantized_params[full_name] = params

                # Replace with fake-quantized weights for downstream layers
                W_quant = self._dequantize_weight(params)
                linear.weight.data = W_quant.to(linear.weight.device, linear.weight.dtype)

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        elapsed = time.time() - start_time
        print("-"*60)
        print(f"Quantization completed in {elapsed:.1f}s")
        print(f"Total quantized layers: {len(self.quantized_params)}")

        return self.quantized_params

    def _quantize_default(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Quantize using default mode with calibration data and Hook-based Input Caching.

        Optimization: Instead of running the full model for each layer (O(L*N) forward passes),
        we cache inputs at the first layer using pre-hooks and propagate them layer-by-layer
        (O(N) forward passes). This reduces quantization time by ~L times (number of layers).

        Returns:
            Dictionary mapping layer names to quantized parameters
        """
        print("="*60)
        print("PT2-LLM: Post-Training Ternarization (Hook-based Optimization)")
        print("="*60)

        start_time = time.time()

        # Get calibration data and layers
        calibration_samples = self.get_calibration_data()
        layers = get_llm_layers(self.model, self.model_type)
        num_layers = len(layers)

        print(f"\nQuantizing {num_layers} transformer layers...")
        print(f"Block size: {self.block_size}, SSR: {self.use_ssr}")
        print("-"*60)

        # =========================================================
        # Phase 1: Cache first layer inputs using pre-hook
        # =========================================================
        print("Phase 1: Caching inputs for the first layer...")
        inps = []
        cache = {'attention_mask': None, 'position_ids': None, 'cache_position': None}

        def input_catcher_hook(module, args, kwargs):
            """Pre-hook to capture inputs before the first layer executes."""
            # Capture hidden states (usually args[0] or kwargs['hidden_states'])
            if len(args) > 0:
                inps.append(args[0].detach())
            elif 'hidden_states' in kwargs:
                inps.append(kwargs['hidden_states'].detach())
            else:
                print(f"Warning: Could not find hidden_states. kwargs keys: {list(kwargs.keys())}")
                return

            # Cache auxiliary parameters (only first time)
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs.get('attention_mask')
            if cache['position_ids'] is None:
                cache['position_ids'] = kwargs.get('position_ids')
            if cache['cache_position'] is None:
                cache['cache_position'] = kwargs.get('cache_position')

            # Interrupt forward pass
            print(".", end="", flush=True)
            raise StopForwardException()

        # Register pre-hook on first layer (works regardless of device_map)
        handle = layers[0].register_forward_pre_hook(input_catcher_hook, with_kwargs=True)

        # Run forward passes (will be interrupted at first layer by hook)
        self.model.eval()
        with torch.no_grad():
            for sample in calibration_samples:
                try:
                    sample = sample.to(self.device)
                    self.model(sample)
                except StopForwardException:
                    pass  # Expected interruption

        # Remove hook to restore normal model behavior
        handle.remove()
        print(f"\nSuccessfully cached {len(inps)} inputs.")

        # Validate that we captured inputs
        if len(inps) == 0:
            raise RuntimeError(
                "CRITICAL: Input caching failed! The hook was not triggered. "
                "Check model structure and get_llm_layers() implementation."
            )

        # =========================================================
        # Phase 2: Layer-wise quantization with input propagation
        # =========================================================
        print(f"Phase 2: Processing {num_layers} layers sequentially...")

        attention_mask = cache['attention_mask']
        position_ids = cache['position_ids']
        cache_position = cache['cache_position']

        for layer_idx in tqdm(range(num_layers), desc="Quantizing Layers"):
            # Move current layer to compute device
            layer = layers[layer_idx].to(self.device)

            # Find linear layers and prepare to capture their inputs
            linear_layers = find_linear_layers(layer)
            subset_inputs = {name: [] for name in linear_layers}

            # Register hooks to capture inputs to each linear layer
            def get_inner_hook(name):
                def hook(module, inp, out):
                    # inp[0] is (Batch, Seq, Dim)
                    subset_inputs[name].append(inp[0].detach())
                return hook

            handles = []
            for name, linear in linear_layers.items():
                handles.append(linear.register_forward_hook(get_inner_hook(name)))

            # Forward pass through current layer, collecting linear inputs
            new_inps = []
            with torch.no_grad():
                for inp in inps:
                    inp = inp.to(self.device)

                    # Build layer kwargs
                    layer_kwargs = {}
                    if attention_mask is not None:
                        layer_kwargs['attention_mask'] = attention_mask
                    if position_ids is not None:
                        layer_kwargs['position_ids'] = position_ids
                    if cache_position is not None:
                        layer_kwargs['cache_position'] = cache_position

                    # Run single layer forward
                    out = layer(inp, **layer_kwargs)

                    # Handle tuple outputs
                    if isinstance(out, tuple):
                        out = out[0]

                    new_inps.append(out)

            # Remove hooks
            for h in handles:
                h.remove()

            # Update inputs for next layer
            inps = new_inps

            # Quantize each linear layer in this transformer layer
            for name, linear in linear_layers.items():
                full_name = f"layer_{layer_idx}.{name}"

                if len(subset_inputs[name]) > 0:
                    # Concatenate all samples: (Total_Tokens, Dim)
                    X = torch.cat(subset_inputs[name], dim=0)

                    # Run quantization (GPTQ/SSR/ATQ)
                    params = self.quantize_layer(linear, full_name, X)
                    self.quantized_params[full_name] = params

                    # Replace with fake-quantized weights for error calibration
                    W_quant = self._dequantize_weight(params)
                    linear.weight.data = W_quant.to(linear.weight.device, linear.weight.dtype)

                    # Free memory
                    del X
                    subset_inputs[name] = None

            # Memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
    parser.add_argument('--dynamic_ssr', action='store_true',
                        help='Use dynamic SSR instead of precomputed K-Means SSR (slower but original method)')
    parser.add_argument('--window_size', type=int, default=WINDOW_SIZE,
                        help=f'Window size for GPTQ error propagation (default: {WINDOW_SIZE})')
    parser.add_argument('--percdamp', type=float, default=0.01,
                        help='GPTQ dampening factor (default: 0.01)')
    parser.add_argument('--mode', type=str, default='default', choices=['default', 'siph'],
                        help='Quantization mode: "default" (with calibration data) or "siph" (data-free) (default: default)')
    parser.add_argument('--siph_gamma', type=float, default=0.5,
                        help='SIPH uniqueness exponent (only used when mode=siph) (default: 0.5)')

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
        use_precomputed_ssr=not args.dynamic_ssr,
        window_size=args.window_size,
        percdamp=args.percdamp,
        seed=args.seed,
        device=args.device,
        mode=args.mode,
        siph_gamma=args.siph_gamma
    )
    
    # Quantize
    quantized_params = quantizer.quantize()

    # Compute real compressed size (ternary parameters only)
    compressed_size, size_breakdown = compute_compressed_size(quantized_params)
    packed_size = compute_packed_size(quantized_params)

    # Count total weights
    total_weights = sum(p['T'].numel() for p in quantized_params.values())
    bits_per_weight = (compressed_size * 1024**3 * 8) / total_weights

    print(f"\n{'='*60}")
    print("Compression Results:")
    print(f"{'='*60}")
    print(f"Original model size:     {orig_size:.2f} GB (FP16)")
    print(f"Compressed size (int8):  {compressed_size:.3f} GB")
    print(f"Compressed size (2-bit): {packed_size:.3f} GB (theoretical)")
    print(f"Compression ratio:       {orig_size/compressed_size:.1f}x (int8) / {orig_size/packed_size:.1f}x (2-bit)")
    print(f"Bits per weight:         {bits_per_weight:.2f} (int8) / ~1.58 (2-bit packed)")
    print(f"\nSize breakdown:")
    print(f"  T (ternary):  {size_breakdown['T']/(1024**3):.3f} GB")
    print(f"  alpha:        {size_breakdown['alpha']/(1024**3):.4f} GB")
    print(f"  mu:           {size_breakdown['mu']/(1024**3):.4f} GB")
    print(f"  perm:         {size_breakdown['perm']/(1024**3):.4f} GB")
    
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
