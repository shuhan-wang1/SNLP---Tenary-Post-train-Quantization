"""
PT2-LLM: GPTQ Framework Integration

Integrates ATQ and SSR within the GPTQ block-wise quantization framework.
GPTQ provides Hessian-guided error compensation after each block.

Reference: 
- GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers
- PT2-LLM Section 3.3 (Efficient Integration with GPTQ)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
import math

from quantizer import AsymmetricTernaryQuantizer
from reorder import SSRReorderer, select_next_block_ssr, compute_column_similarity_to_mean, get_ssr_permutation

# Windowed error propagation: only update nearby columns for O(NMK) instead of O(NM^2)
# Based on Hessian's diagonal-dominant structure after SSR reordering
WINDOW_SIZE = 2048


class GPTQ:
    """
    GPTQ quantization with PT2-LLM's ATQ and SSR integration.
    
    Performs block-wise quantization with:
    1. SSR-based column reordering for homogeneous blocks
    2. ATQ for ternary quantization
    3. Hessian-guided error compensation
    """
    
    def __init__(self, layer: nn.Linear, block_size: int = 128, percdamp: float = 0.01,
                 window_size: int = WINDOW_SIZE):
        """
        Args:
            layer: Linear layer to quantize
            block_size: Number of columns per quantization block
            percdamp: Dampening factor for Hessian inverse
            window_size: Window size for error propagation (default 2048)
        """
        self.layer = layer
        self.block_size = block_size
        self.percdamp = percdamp
        self.window_size = window_size

        self.device = layer.weight.device
        self.dtype = layer.weight.dtype

        # Weight shape: (out_features, in_features)
        W = layer.weight.data.clone()
        self.rows, self.columns = W.shape

        # Initialize Hessian
        self.H = torch.zeros((self.columns, self.columns), device=self.device, dtype=self.dtype)
        self.nsamples = 0

        # Storage for quantized parameters
        self.alpha = None
        self.mu = None
        self.T = None
        self.perm = None
        
    def add_batch(self, inp: torch.Tensor):
        """
        Accumulate Hessian approximation from input batch.
        
        H = 2 * X^T X (for MSE loss)
        
        Args:
            inp: Input activations of shape (batch, seq_len, in_features) or (batch, in_features)
        """
        if inp.dim() == 3:
            inp = inp.reshape(-1, inp.shape[-1])
        
        batch_size = inp.shape[0]
        
        # Accumulate H = X^T X
        inp = inp.t()  # (in_features, batch)
        self.H += inp @ inp.t()
        self.nsamples += batch_size
    
    def quantize(self, use_ssr: bool = True, use_precomputed_ssr: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Perform PT2-LLM quantization with GPTQ error compensation.

        Args:
            use_ssr: Whether to use SSR column reordering
            use_precomputed_ssr: If True, use K-Means precomputed permutation (faster);
                                 If False, use dynamic block selection (original method)

        Returns:
            alpha: Scaling factors for each block, shape (rows, num_blocks)
            mu: Offsets for each block, shape (rows, num_blocks)
            T: Ternary matrix, shape (rows, columns)
            perm: Column permutation indices
        """
        W = self.layer.weight.data.clone()

        # Normalize Hessian
        H = self.H / self.nsamples

        # Add dampening for numerical stability
        damp = self.percdamp * torch.diag(H).mean()
        damp = max(damp.item(), 1e-4)  # Ensure minimum damping to prevent ill-conditioning
        H.diagonal().add_(damp)

        # Compute Hessian inverse using Cholesky decomposition
        try:
            L = torch.linalg.cholesky(H)
            H_inv = torch.cholesky_inverse(L)
            H_inv = torch.nan_to_num(H_inv, nan=0.0, posinf=0.0, neginf=0.0)
        except RuntimeError:
            # Fallback to pseudo-inverse if Cholesky fails
            H_inv = torch.linalg.pinv(H)
            H_inv = torch.nan_to_num(H_inv, nan=0.0, posinf=0.0, neginf=0.0)

        # Initialize output tensors
        T_full = torch.zeros_like(W)
        alpha_blocks = []
        mu_blocks = []

        # ========== SSR Reordering Strategy ==========
        if use_ssr:
            if use_precomputed_ssr:
                # Optimized: Precompute full permutation using K-Means clustering
                # O(m*k*iter) instead of O(m^2) for dynamic SSR
                perm = get_ssr_permutation(W, self.block_size)
                # Reorder W and H_inv according to permutation
                W = W[:, perm]
                H_inv = H_inv[perm][:, perm]
                H = H[perm][:, perm]
            else:
                # Original: Dynamic block selection (slower)
                remaining_indices = torch.arange(self.columns, device=self.device)
                perm_list = []
        else:
            perm = torch.arange(self.columns, device=self.device)

        # ATQ quantizer
        atq = AsymmetricTernaryQuantizer()

        # Process blocks sequentially (after reordering, just iterate in order)
        col_idx = 0
        while col_idx < self.columns:
            # Determine block columns
            if use_ssr and not use_precomputed_ssr:
                # Dynamic SSR: Select columns most similar to remaining mean
                block_indices, remaining_indices = select_next_block_ssr(
                    W, remaining_indices, self.block_size
                )
                perm_list.extend(block_indices.tolist())
                block_size = len(block_indices)
            else:
                # Sequential order (either no SSR or precomputed SSR already reordered W)
                end_idx = min(col_idx + self.block_size, self.columns)
                block_indices = torch.arange(col_idx, end_idx, device=self.device)
                block_size = len(block_indices)

            if block_size == 0:
                break

            # Extract block weights and corresponding Hessian elements
            W_block = W[:, block_indices]

            # Collect covariance submatrix for this block
            # X_block is S[block, block] = (X^T X)[block, block], already a covariance matrix
            S_block = H[:, block_indices][block_indices, :]  # Hessian submatrix (covariance)

            # Apply ATQ to block (ITF + AGA)
            # FIX: Pass as S parameter directly to avoid computing S = X.T @ X again
            # Previously X_block was passed as X, causing AGA to compute (X^T X)^T (X^T X)
            alpha_b, mu_b, T_b = atq.quantize(W_block, X=None, S=S_block)

            # Store block results
            alpha_blocks.append(alpha_b)
            mu_blocks.append(mu_b)
            T_full[:, block_indices] = T_b

            # Compute quantization error for this block
            W_quant_block = alpha_b * T_b + mu_b
            quant_error = W_block - W_quant_block

            # ========== GPTQ Error Compensation with Windowing ==========
            # Remaining column indices
            if use_ssr and not use_precomputed_ssr:
                remaining_cols = remaining_indices
            else:
                remaining_cols = torch.arange(col_idx + block_size, self.columns, device=self.device)

            if len(remaining_cols) > 0:
                # Windowed update: only update the nearest WINDOW_SIZE columns
                # This exploits the diagonal-dominant structure of H_inv after SSR reordering
                # Reduces complexity from O(NM^2) to O(NMK) where K = window_size
                limit = min(len(remaining_cols), self.window_size)
                update_cols = remaining_cols[:limit]

                # 1. Extract H_inv[block, update_cols]
                H_inv_block_rem = H_inv[block_indices][:, update_cols]

                # 2. Extract diagonal elements
                H_inv_diag = H_inv[block_indices, block_indices].unsqueeze(1).clamp(min=1e-8)

                # 3. Compute coefficients with clamping to prevent numerical explosion
                coefficients = H_inv_block_rem / H_inv_diag
                coefficients = coefficients.clamp(-10.0, 10.0)

                # 4. Clamp quantization error to prevent explosion
                quant_error_clamped = quant_error.clamp(-10.0, 10.0)

                # 5. Vectorized update with safety checks
                update = quant_error_clamped @ coefficients

                # Skip update if NaN or Inf detected to prevent error propagation
                if not (torch.isnan(update).any() or torch.isinf(update).any()):
                    W[:, update_cols] -= update
                    # Clamp W to prevent cumulative explosion across blocks
                    W[:, update_cols] = W[:, update_cols].clamp(-100.0, 100.0)

            col_idx += block_size

        # Finalize permutation
        if use_ssr and not use_precomputed_ssr:
            perm = torch.tensor(perm_list, device=self.device)

        # Stack block parameters
        self.alpha = torch.cat(alpha_blocks, dim=1)  # (rows, num_blocks)
        self.mu = torch.cat(mu_blocks, dim=1)        # (rows, num_blocks)
        self.T = T_full
        self.perm = perm

        return self.alpha, self.mu, self.T, perm
    
    def get_quantized_weight(self) -> torch.Tensor:
        """Get dequantized weight matrix."""
        if self.T is None:
            raise RuntimeError("Must call quantize() first")
        
        # Reconstruct from stored parameters
        # Note: alpha and mu are stored per-block, need to expand
        W_quant = torch.zeros(self.rows, self.columns, device=self.device, dtype=self.dtype)
        
        num_blocks = self.alpha.shape[1]
        block_idx = 0
        
        for b in range(num_blocks):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, self.columns)
            actual_block_size = end - start
            
            alpha_b = self.alpha[:, b:b+1]  # (rows, 1)
            mu_b = self.mu[:, b:b+1]         # (rows, 1)
            
            # Get columns for this block (considering permutation)
            if self.perm is not None:
                block_cols = self.perm[start:end]
            else:
                block_cols = torch.arange(start, end, device=self.device)
            
            T_block = self.T[:, block_cols]
            W_quant[:, block_cols] = alpha_b * T_block + mu_b
        
        return W_quant


class GPTQQuantizer:
    """
    High-level GPTQ quantizer for full model quantization.
    """
    
    def __init__(self, 
                 model: nn.Module,
                 block_size: int = 128,
                 percdamp: float = 0.01,
                 use_ssr: bool = True):
        self.model = model
        self.block_size = block_size
        self.percdamp = percdamp
        self.use_ssr = use_ssr
        
        self.quantizers: Dict[str, GPTQ] = {}
        self.calibration_data = []
        
    def add_calibration_data(self, data: torch.Tensor):
        """Add calibration data batch."""
        self.calibration_data.append(data)
    
    def prepare_quantizer(self, name: str, layer: nn.Linear):
        """Create GPTQ quantizer for a layer."""
        self.quantizers[name] = GPTQ(layer, self.block_size, self.percdamp)
    
    def quantize_layer(self, name: str) -> Dict[str, torch.Tensor]:
        """Quantize a prepared layer."""
        if name not in self.quantizers:
            raise ValueError(f"Layer {name} not prepared for quantization")
        
        gptq = self.quantizers[name]
        alpha, mu, T, perm = gptq.quantize(use_ssr=self.use_ssr)
        
        return {
            'alpha': alpha,
            'mu': mu,
            'T': T,
            'perm': perm
        }
