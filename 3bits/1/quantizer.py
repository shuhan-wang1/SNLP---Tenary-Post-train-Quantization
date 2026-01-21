"""
PT2-LLM: Asymmetric Ternary Quantizer (ATQ)
Implements:
- Asymmetric Ternary Initialization
- Iterative Ternary Fitting (ITF)
- Activation-aware Grid Alignment (AGA)

Reference: PT2-LLM: Post-Training Ternarization for Large Language Models
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional


class AsymmetricTernaryQuantizer:
    """
    Asymmetric Ternary Quantizer with two-stage refinement:
    1. Iterative Ternary Fitting (ITF): Alternates between optimal grid construction
       and flexible rounding to minimize quantization error
    2. Activation-aware Grid Alignment (AGA): Refines ternary grid to match 
       full-precision outputs using calibration data
    """
    
    def __init__(self, max_iter: int = 5):
        """
        Args:
            max_iter: Maximum iterations for ITF convergence (default 5,
                      based on convergence analysis showing ~95% improvement in first 5 iterations)
        """
        self.max_iter = max_iter
    
    def ternary_init(self, W: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Asymmetric Ternary Initialization (Section 3.2)
        
        Initializes ternary parameters with row-wise offset to handle non-zero mean distributions.
        
        Args:
            W: Weight matrix of shape (n, m)
            
        Returns:
            alpha: Row-wise scaling factor of shape (n, 1)
            mu: Row-wise offset of shape (n, 1)
            T: Ternary matrix of shape (n, m) with values in {-1, 0, 1}
        """
        n, m = W.shape
        
        # Eq. 4: Compute row-wise mean as offset
        mu = W.mean(dim=1, keepdim=True)  # (n, 1)
        
        # Center the weights
        W_centered = W - mu  # W_f = W - μ
        
        # Eq. 5: Compute threshold using TWN approximation
        # Δ ≈ 0.75/m * Σ|W_f_{:,j}|
        delta = 0.75 * W_centered.abs().mean(dim=1, keepdim=True)  # (n, 1)
        
        # Eq. 2: Threshold-based ternary assignment
        T = torch.zeros_like(W)
        T[W_centered > delta] = 1.0
        T[W_centered < -delta] = -1.0
        
        # Eq. 5: Compute optimal scaling factor
        # α = Σ(T_{:,j} · W_f_{:,j}) / Σ|T_{:,j}|
        numerator = (T * W_centered).sum(dim=1, keepdim=True)
        denominator = T.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
        alpha = numerator / denominator  # (n, 1)
        
        return alpha, mu, T
    
    def build_optimal_grid(self, W: torch.Tensor, T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        n, m = W.shape

        WT_sum = (W * T).sum(dim=1, keepdim=True)
        T_sum = T.sum(dim=1, keepdim=True)
        W_sum = W.sum(dim=1, keepdim=True)
        T2_sum = (T * T).sum(dim=1, keepdim=True)

        denominator = m * T2_sum - T_sum ** 2

        # Degenerate detection: T is all zeros or nearly constant
        degenerate = (denominator.abs() < 1e-6)
        denominator = torch.where(degenerate, torch.ones_like(denominator), denominator)

        alpha = (m * WT_sum - T_sum * W_sum) / denominator
        mu = (T2_sum * W_sum - T_sum * WT_sum) / denominator

        # Degenerate rows: use simple fallback (alpha=0, mu=row_mean)
        row_mean = W.mean(dim=1, keepdim=True)
        alpha = torch.where(degenerate, torch.zeros_like(alpha), alpha)
        mu = torch.where(degenerate, row_mean, mu)

        # Clamp to prevent Inf at the source
        alpha = alpha.clamp(-100.0, 100.0)
        mu = mu.clamp(-100.0, 100.0)

        return alpha, mu
    
    def flexible_round(self, W: torch.Tensor, alpha: torch.Tensor, mu: torch.Tensor) -> torch.Tensor:
        """
        Flexible Rounding (Eq. 10 in Section 3.2)
        
        Updates T by mapping weights to nearest ternary value given current grid.
        
        Args:
            W: Original weight matrix of shape (n, m)
            alpha: Scaling factor of shape (n, 1)
            mu: Offset of shape (n, 1)
            
        Returns:
            T: Updated ternary matrix of shape (n, m)
        """
        # Eq. 10: Z_ij = (W_ij - μ*_i) / α*_i
        alpha_safe = alpha.clamp(min=1e-8)
        Z = (W - mu) / alpha_safe
        
        # T*_ij = argmin_{t∈{-1,0,1}} |Z_ij - t|
        # Boundaries are at -0.5 and 0.5
        T = torch.zeros_like(W)
        T[Z > 0.5] = 1.0
        T[Z < -0.5] = -1.0
        
        return T
    
    def iterative_ternary_fitting(self, W: torch.Tensor, 
                                   alpha: torch.Tensor, 
                                   mu: torch.Tensor, 
                                   T: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Iterative Ternary Fitting (ITF) - Section 3.2
        
        Alternates between:
        1. Optimal Grid Construction: Find best α, μ for current T
        2. Flexible Rounding: Find best T for current α, μ
        
        Continues until T stabilizes (convergence).
        
        Args:
            W: Original weight matrix
            alpha: Initial scaling factor
            mu: Initial offset
            T: Initial ternary matrix
            
        Returns:
            alpha: Refined scaling factor
            mu: Refined offset
            T: Refined ternary matrix
        """
        T_prev = torch.zeros_like(T)
        
        for iteration in range(self.max_iter):
            # Check convergence: T unchanged
            if torch.equal(T, T_prev):
                break
            
            T_prev = T.clone()
            
            # Step 1: Build optimal grid for current T
            alpha, mu = self.build_optimal_grid(W, T)
            
            # Step 2: Update T via flexible rounding
            T = self.flexible_round(W, alpha, mu)
        
        return alpha, mu, T
    
    def activation_aware_grid_alignment(self, W: torch.Tensor,
                                     T: torch.Tensor,
                                     X: Optional[torch.Tensor] = None,
                                     S: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        n, m = W.shape

        if S is None:
            if X is None:
                raise ValueError("Either X or S must be provided for AGA")
            if X.dim() == 3:
                X = X.reshape(-1, m)
            S = X.T @ X

        ones = torch.ones(m, 1, device=W.device, dtype=W.dtype)
        S1 = S @ ones

        d = (ones.T @ S1).clamp(min=1e-8).item()  # Prevent d from being 0
        v = T @ S1
        WS1 = W @ S1
        WT_S1 = (W * T) @ S1
        T2 = T * T
        T2_S1 = T2 @ S1
        v2 = v * v

        denominator = d * T2_S1 - v2

        # Degenerate detection
        degenerate = (denominator.abs() < 1e-6)
        denominator = torch.where(degenerate, torch.ones_like(denominator), denominator)

        alpha = (d * WT_S1 - v * WS1) / denominator
        mu = (T2_S1 * WS1 - v * WT_S1) / denominator

        # Degenerate rows: fallback to ITF result
        alpha_itf, mu_itf = self.build_optimal_grid(W, T)
        alpha = torch.where(degenerate, alpha_itf, alpha)
        mu = torch.where(degenerate, mu_itf, mu)

        # nan_to_num + clamp to ensure no Inf/NaN escapes
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=10.0, neginf=-10.0)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=10.0, neginf=-10.0)
        alpha = alpha.clamp(-100.0, 100.0)
        mu = mu.clamp(-100.0, 100.0)

        return alpha, mu
    
    def quantize(self, W: torch.Tensor, X: Optional[torch.Tensor] = None,
             S: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        alpha, mu, T = self.ternary_init(W)
        alpha, mu, T = self.iterative_ternary_fitting(W, alpha, mu, T)

        if X is not None or S is not None:
            alpha, mu = self.activation_aware_grid_alignment(W, T, X=X, S=S)

        # Final safety check: nan_to_num + clamp
        alpha = torch.nan_to_num(alpha, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-100, 100)
        mu = torch.nan_to_num(mu, nan=0.0, posinf=10.0, neginf=-10.0).clamp(-100, 100)

        return alpha, mu, T
    
    def dequantize(self, alpha: torch.Tensor, mu: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        """
        Dequantize ternary representation back to approximate weights.
        
        W_c = αT + μ
        
        Args:
            alpha: Scaling factor, shape (n, 1)
            mu: Offset, shape (n, 1)
            T: Ternary matrix, shape (n, m)
            
        Returns:
            W_c: Dequantized weight approximation, shape (n, m)
        """
        return alpha * T + mu


def compute_siph_hessian(W: torch.Tensor, block_size: int, gamma: float = 0.5) -> torch.Tensor:
    """
    Computes the SSR-Induced Pseudo-Hessian (SIPH) matrix.

    This is a data-free sensitivity proxy that synthesizes a "Pseudo-Hessian" matrix
    purely from weight statistics, allowing GPTQ-style error compensation without
    any calibration data.

    Mathematical Formula for diagonal elements:
    H_jj = ||W_{:,j}||_2^2 * (1 + Var(W_{:,j})/MeanVar_b) * (1 - avg_sim_j)^(-gamma)

    Where:
    - ||W_{:,j}||_2^2: Column magnitude (L2 norm squared)
    - Var(W_{:,j})/MeanVar_b: Relative variance within block
    - avg_sim_j: Average cosine similarity of column j to other columns in block
    - gamma: Uniqueness exponent (default 0.5)

    Returns a dense approximation: H_ij = sqrt(H_ii * H_jj) * Similarity_ij

    Args:
        W: Weight matrix of shape (n, m), where n=out_features, m=in_features
        block_size: Block size for computing local statistics
        gamma: Exponent for uniqueness term (default 0.5)

    Returns:
        H_pseudo: Pseudo-Hessian matrix of shape (m, m)
    """
    n, m = W.shape
    device = W.device
    dtype = W.dtype

    # 1. Compute Cosine Similarity Matrix S (m x m)
    # S_ij = cos(W_{:,i}, W_{:,j}) = (W_{:,i}^T W_{:,j}) / (||W_{:,i}|| ||W_{:,j}||)
    col_norms = W.norm(dim=0, keepdim=True).clamp(min=1e-8)  # (1, m)
    W_norm = W / col_norms  # Normalized columns
    S = W_norm.T @ W_norm  # (m, m) cosine similarity matrix

    # 2. Compute per-column statistics
    col_norms_sq = (col_norms.squeeze() ** 2)  # (m,) - L2 norm squared
    col_vars = W.var(dim=0)  # (m,) - variance of each column

    # 3. Compute Diagonal H_ii terms block-by-block
    H_diag = torch.zeros(m, device=device, dtype=dtype)

    for start in range(0, m, block_size):
        end = min(start + block_size, m)
        block_len = end - start

        # Block statistics
        block_vars = col_vars[start:end]
        mean_block_var = block_vars.mean()

        # Intra-block similarity (exclude self-similarity of 1.0)
        S_block = S[start:end, start:end]

        # Average similarity to other columns in block: (row_sum - 1) / (block_len - 1)
        # Handles edge case where block has only 1 column
        if block_len > 1:
            avg_sim = (S_block.sum(dim=1) - 1) / (block_len - 1)
        else:
            avg_sim = torch.zeros(1, device=device, dtype=dtype)

        # Calculate SIPH diagonal components
        # Magnitude: ||W_{:,j}||_2^2
        magnitude = col_norms_sq[start:end]

        # Relative variance: 1 + Var(W_{:,j}) / MeanVar_b
        rel_var = 1 + (block_vars / (mean_block_var + 1e-8))

        # Uniqueness: (1 - avg_sim)^(-gamma)
        # Clamp avg_sim to avoid division by zero when avg_sim = 1
        uniqueness = (1 - avg_sim.clamp(max=0.99)) ** (-gamma)

        # H_jj = magnitude * rel_var * uniqueness
        H_diag[start:end] = magnitude * rel_var * uniqueness

    # 4. Construct Dense Pseudo-Hessian
    # H_ij ≈ sqrt(H_ii) * sqrt(H_jj) * S_ij
    # This captures the covariance structure while incorporating sensitivity estimates
    sqrt_diag = torch.sqrt(H_diag).unsqueeze(1)  # (m, 1)
    H_pseudo = (sqrt_diag @ sqrt_diag.T) * S  # (m, m)

    return H_pseudo


def compute_quantization_error(W: torch.Tensor, W_c: torch.Tensor) -> float:
    """Compute quantization error E_w = ||W - W_c||²_F"""
    return ((W - W_c) ** 2).sum().item()


def compute_output_error(W: torch.Tensor, W_c: torch.Tensor, X: torch.Tensor) -> float:
    """Compute output error E_x = ||WX - W_cX||²_F"""
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    diff = (W - W_c) @ X.T
    return (diff ** 2).sum().item()
