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
    
    def __init__(self, max_iter: int = 100):
        """
        Args:
            max_iter: Maximum iterations for ITF convergence
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
        """
        Optimal Grid Construction (Eq. 9 in Section 3.2)
        
        Given fixed T, computes the optimal α* and μ* that minimize quantization error E_w.
        Uses closed-form solution derived from setting partial derivatives to zero.
        
        Args:
            W: Original weight matrix of shape (n, m)
            T: Current ternary matrix of shape (n, m)
            
        Returns:
            alpha: Optimal scaling factor of shape (n, 1)
            mu: Optimal offset of shape (n, 1)
        """
        n, m = W.shape
        
        # Vectorized computation of Eq. 9:
        # α* = (m · (W ◦ T)1 - (T1) ◦ (W1)) / (m · (T ◦ T)1 - (T1)²)
        # μ* = ((T ◦ T)1 ◦ (W1) - (T1) ◦ [(W ◦ T)1]) / (m · (T ◦ T)1 - (T1)²)
        
        # Element-wise products summed along rows
        WT_sum = (W * T).sum(dim=1, keepdim=True)      # (W ◦ T)1
        T_sum = T.sum(dim=1, keepdim=True)              # T1
        W_sum = W.sum(dim=1, keepdim=True)              # W1
        T2_sum = (T * T).sum(dim=1, keepdim=True)      # (T ◦ T)1
        
        # Denominator: m · (T ◦ T)1 - (T1)²
        denominator = m * T2_sum - T_sum ** 2
        denominator = denominator.clamp(min=1e-8)  # Numerical stability
        
        # Optimal alpha: α* = (m · (W ◦ T)1 - (T1) ◦ (W1)) / denominator
        alpha = (m * WT_sum - T_sum * W_sum) / denominator
        
        # Optimal mu: μ* = ((T ◦ T)1 ◦ (W1) - (T1) ◦ [(W ◦ T)1]) / denominator
        mu = (T2_sum * W_sum - T_sum * WT_sum) / denominator
        
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
                                         X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Activation-aware Grid Alignment (AGA) - Section 3.2
        
        Refines α and μ to minimize output error E_x = ||WX - W_cX||²_F
        where W_c = αT + μ.
        
        Uses calibration activations to align quantized outputs with full-precision.
        
        Args:
            W: Original weight matrix of shape (n, m)
            T: Ternary matrix of shape (n, m)
            X: Calibration activations of shape (B*L, m) or reshaped appropriately
            
        Returns:
            alpha: Activation-aligned scaling factor
            mu: Activation-aligned offset
        """
        n, m = W.shape
        
        # Compute covariance matrix C = Σ X_i X_i^T
        # For efficiency, we compute S = X^T X which gives us what we need
        # C_ij = Σ_k X_ki * X_kj
        if X.dim() == 3:
            # X is (B, L, m), reshape to (B*L, m)
            X = X.reshape(-1, m)
        
        # S = X^T X, shape (m, m) - this is the covariance matrix
        S = X.T @ X  # (m, m)
        
        # Eq. 13: Compute optimal α* and μ* using activation-aware formulation
        # d = 1^T S 1 (scalar)
        # v = TS1 (n, 1)
        # α* = (d · (W ◦ T)S1 - v ◦ (WS1)) / (d · T²S1 - v²)
        # μ* = (T²S1 ◦ (WS1) - v ◦ [(W ◦ T)S1]) / (d · T²S1 - v²)
        
        ones = torch.ones(m, 1, device=W.device, dtype=W.dtype)
        S1 = S @ ones  # (m, 1)
        
        d = (ones.T @ S1).item()  # scalar: 1^T S 1
        
        # v = TS1, shape (n, 1)
        v = T @ S1  # (n, 1)
        
        # WS1, shape (n, 1)
        WS1 = W @ S1  # (n, 1)
        
        # (W ◦ T)S1, shape (n, 1)
        WT_S1 = (W * T) @ S1  # (n, 1)
        
        # T² element-wise squared
        T2 = T * T  # (n, m)
        
        # T²S1, shape (n, 1)
        T2_S1 = T2 @ S1  # (n, 1)
        
        # v², element-wise square
        v2 = v * v  # (n, 1)
        
        # Denominator: d · T²S1 - v²
        denominator = d * T2_S1 - v2
        denominator = denominator.clamp(min=1e-8)  # Numerical stability
        
        # Optimal alpha: α* = (d · (W ◦ T)S1 - v ◦ (WS1)) / denominator
        alpha = (d * WT_S1 - v * WS1) / denominator
        
        # Optimal mu: μ* = (T²S1 ◦ (WS1) - v ◦ [(W ◦ T)S1]) / denominator
        mu = (T2_S1 * WS1 - v * WT_S1) / denominator
        
        return alpha, mu
    
    def quantize(self, W: torch.Tensor, X: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full ATQ quantization pipeline (Algorithm 1)
        
        1. Ternary initialization with asymmetric offset
        2. Iterative Ternary Fitting to minimize weight error
        3. Activation-aware Grid Alignment to minimize output error
        
        Args:
            W: Weight matrix to quantize, shape (n, m)
            X: Calibration activations, shape (B, L, m) or (B*L, m)
            
        Returns:
            alpha: Final scaling factor, shape (n, 1)
            mu: Final offset, shape (n, 1)
            T: Final ternary matrix, shape (n, m)
        """
        # Step 1: Asymmetric ternary initialization
        alpha, mu, T = self.ternary_init(W)
        
        # Step 2: Iterative Ternary Fitting (ITF)
        alpha, mu, T = self.iterative_ternary_fitting(W, alpha, mu, T)
        
        # Step 3: Activation-aware Grid Alignment (AGA)
        if X is not None:
            alpha, mu = self.activation_aware_grid_alignment(W, T, X)
        
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


def compute_quantization_error(W: torch.Tensor, W_c: torch.Tensor) -> float:
    """Compute quantization error E_w = ||W - W_c||²_F"""
    return ((W - W_c) ** 2).sum().item()


def compute_output_error(W: torch.Tensor, W_c: torch.Tensor, X: torch.Tensor) -> float:
    """Compute output error E_x = ||WX - W_cX||²_F"""
    if X.dim() == 3:
        X = X.reshape(-1, X.shape[-1])
    diff = (W - W_c) @ X.T
    return (diff ** 2).sum().item()
