"""
PT2-LLM: Structural Similarity-based Reordering (SSR)

Reorders weight columns based on structural similarity to:
1. Create more homogeneous blocks for ternarization
2. Group outliers together to reduce their distorting effect

Reference: PT2-LLM Section 3.3
"""

import torch
from typing import Tuple, List


def compute_cosine_similarity_matrix(W: torch.Tensor) -> torch.Tensor:
    """
    Compute pairwise cosine similarity between weight columns (Eq. 15).
    
    S_ij = (W_{:,i}^T W_{:,j}) / (||W_{:,i}||_2 ||W_{:,j}||_2)
    
    Args:
        W: Weight matrix of shape (n, m)
        
    Returns:
        S: Similarity matrix of shape (m, m)
    """
    # Normalize columns
    W_norm = W / (W.norm(dim=0, keepdim=True).clamp(min=1e-8))
    
    # Compute cosine similarity: S = W_norm^T @ W_norm
    S = W_norm.T @ W_norm
    
    return S


def compute_column_similarity_to_mean(W: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """
    Compute similarity of remaining columns to their mean vector (Eq. 16).
    
    Used for efficient integration with GPTQ's block-by-block quantization.
    
    Args:
        W: Weight matrix of shape (n, m)
        indices: Indices of remaining unquantized columns
        
    Returns:
        similarities: Similarity scores for each remaining column
    """
    W_remaining = W[:, indices]
    
    # Compute mean vector: w̄ = (1/m) Σ W_{:,i}
    w_mean = W_remaining.mean(dim=1, keepdim=True)  # (n, 1)
    
    # Normalize
    w_mean_norm = w_mean / (w_mean.norm().clamp(min=1e-8))
    W_remaining_norm = W_remaining / (W_remaining.norm(dim=0, keepdim=True).clamp(min=1e-8))
    
    # Compute similarities: (W_{:,i}^T w̄) / (||W_{:,i}||_2 ||w̄||_2)
    similarities = (W_remaining_norm.T @ w_mean_norm).squeeze()
    
    return similarities


def get_initial_reorder_indices(W: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Get initial column reordering based on full similarity matrix.
    
    Uses greedy clustering: start with most central column, then iteratively
    add columns most similar to current cluster mean.
    
    Args:
        W: Weight matrix of shape (n, m)
        block_size: Size of quantization blocks
        
    Returns:
        perm: Permutation indices for columns
    """
    n, m = W.shape
    S = compute_cosine_similarity_matrix(W)
    
    # Track which columns have been selected
    selected = torch.zeros(m, dtype=torch.bool, device=W.device)
    perm = []
    
    # Start with the column most similar to all others (highest row sum)
    row_sums = S.sum(dim=1)
    start_idx = row_sums.argmax().item()
    perm.append(start_idx)
    selected[start_idx] = True
    
    # Greedily add columns
    while len(perm) < m:
        # Find unselected column most similar to current cluster
        current_indices = torch.tensor(perm, device=W.device)
        
        # Compute similarity to cluster (average similarity to selected columns)
        cluster_sim = S[:, current_indices].mean(dim=1)
        cluster_sim[selected] = -float('inf')
        
        next_idx = cluster_sim.argmax().item()
        perm.append(next_idx)
        selected[next_idx] = True
    
    return torch.tensor(perm, device=W.device)


def select_next_block_ssr(W: torch.Tensor, 
                          remaining_indices: torch.Tensor, 
                          block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SSR Block Selection (Eq. 16): Select next block using structural similarity.
    
    After each GPTQ update, selects top-k columns most similar to the mean
    of remaining columns for the next quantization block.
    
    Args:
        W: Current weight matrix (with GPTQ error compensation)
        remaining_indices: Indices of columns not yet quantized
        block_size: Number of columns per block
        
    Returns:
        block_indices: Original indices of columns in this block
        new_remaining: Updated remaining indices
    """
    if len(remaining_indices) <= block_size:
        return remaining_indices, torch.tensor([], dtype=remaining_indices.dtype, device=remaining_indices.device)
    
    # Compute similarities to mean
    similarities = compute_column_similarity_to_mean(W, remaining_indices)
    
    # Select top-k most similar columns
    k = min(block_size, len(remaining_indices))
    _, top_k_local = torch.topk(similarities, k)
    
    # Map back to original indices
    block_indices = remaining_indices[top_k_local]
    
    # Update remaining indices
    mask = torch.ones(len(remaining_indices), dtype=torch.bool, device=remaining_indices.device)
    mask[top_k_local] = False
    new_remaining = remaining_indices[mask]
    
    return block_indices, new_remaining


class SSRReorderer:
    """
    Structural Similarity-based Reordering handler.
    
    Provides both:
    1. Full initial reordering (for preprocessing)
    2. Dynamic block selection (for GPTQ integration)
    """
    
    def __init__(self, W: torch.Tensor, block_size: int = 128, use_dynamic: bool = True):
        """
        Args:
            W: Weight matrix to analyze
            block_size: Quantization block size
            use_dynamic: If True, use dynamic block selection during GPTQ
        """
        self.block_size = block_size
        self.use_dynamic = use_dynamic
        self.n, self.m = W.shape
        
        if not use_dynamic:
            # Precompute full reordering
            self.perm = get_initial_reorder_indices(W, block_size)
            self.inv_perm = torch.argsort(self.perm)
        else:
            # For dynamic selection, start with original order
            self.perm = torch.arange(self.m, device=W.device)
            self.inv_perm = self.perm.clone()
    
    def get_permutation(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get permutation and inverse permutation indices."""
        return self.perm, self.inv_perm
    
    def reorder_weights(self, W: torch.Tensor) -> torch.Tensor:
        """Apply reordering to weight matrix."""
        return W[:, self.perm]
    
    def reorder_activations(self, X: torch.Tensor) -> torch.Tensor:
        """Apply reordering to activation matrix (last dimension)."""
        return X[..., self.perm]
    
    def restore_order(self, W: torch.Tensor) -> torch.Tensor:
        """Restore original column order."""
        return W[:, self.inv_perm]


def apply_permutation(W: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply column permutation to weight matrix: W' = WP"""
    return W[:, perm]


def apply_permutation_to_input(X: torch.Tensor, perm: torch.Tensor) -> torch.Tensor:
    """Apply permutation to input features: X' = XP"""
    if X.dim() == 2:
        return X[:, perm]
    elif X.dim() == 3:
        return X[:, :, perm]
    else:
        raise ValueError(f"Unexpected input dimension: {X.dim()}")


def compute_block_variance(W: torch.Tensor, block_size: int) -> List[float]:
    """
    Compute variance within each block for analysis.
    
    Lower variance indicates more homogeneous blocks, which is better for ternarization.
    """
    n, m = W.shape
    variances = []
    
    for i in range(0, m, block_size):
        end = min(i + block_size, m)
        block = W[:, i:end]
        variances.append(block.var().item())
    
    return variances
