"""
PT2-LLM: Example Usage Script

Demonstrates individual components and their usage.
"""

import torch
import torch.nn as nn

# Import PT2-LLM components
from quantizer import AsymmetricTernaryQuantizer, compute_quantization_error, compute_output_error
from reorder import SSRReorderer, compute_cosine_similarity_matrix, compute_block_variance


def example_atq_basic():
    """Example: Basic ATQ quantization of a weight matrix."""
    print("=" * 50)
    print("Example 1: Basic ATQ Quantization")
    print("=" * 50)
    
    # Create a sample weight matrix
    torch.manual_seed(42)
    W = torch.randn(256, 512)  # (out_features, in_features)
    X = torch.randn(32, 512)   # Calibration activations
    
    # Create quantizer
    atq = AsymmetricTernaryQuantizer(max_iter=100)
    
    # Step 1: Ternary initialization
    alpha_init, mu_init, T_init = atq.ternary_init(W)
    W_init = atq.dequantize(alpha_init, mu_init, T_init)
    err_init = compute_quantization_error(W, W_init)
    print(f"Initial quantization error: {err_init:.4f}")
    
    # Step 2: ITF refinement
    alpha_itf, mu_itf, T_itf = atq.iterative_ternary_fitting(W, alpha_init, mu_init, T_init)
    W_itf = atq.dequantize(alpha_itf, mu_itf, T_itf)
    err_itf = compute_quantization_error(W, W_itf)
    print(f"After ITF error: {err_itf:.4f} (reduction: {100*(1-err_itf/err_init):.1f}%)")
    
    # Step 3: AGA alignment
    alpha_aga, mu_aga = atq.activation_aware_grid_alignment(W, T_itf, X)
    W_aga = atq.dequantize(alpha_aga, mu_aga, T_itf)
    out_err_before = compute_output_error(W, W_itf, X)
    out_err_after = compute_output_error(W, W_aga, X)
    print(f"Output error before AGA: {out_err_before:.4f}")
    print(f"Output error after AGA: {out_err_after:.4f}")
    print()


def example_atq_full():
    """Example: Full ATQ pipeline."""
    print("=" * 50)
    print("Example 2: Full ATQ Pipeline")
    print("=" * 50)
    
    torch.manual_seed(42)
    W = torch.randn(256, 512)
    X = torch.randn(32, 512)
    
    atq = AsymmetricTernaryQuantizer()
    alpha, mu, T = atq.quantize(W, X)
    
    W_quant = atq.dequantize(alpha, mu, T)
    
    # Statistics
    err = compute_quantization_error(W, W_quant)
    num_zero = (T == 0).sum().item()
    num_pos = (T == 1).sum().item()
    num_neg = (T == -1).sum().item()
    total = T.numel()
    
    print(f"Quantization error: {err:.4f}")
    print(f"Ternary distribution: -1={100*num_neg/total:.1f}%, 0={100*num_zero/total:.1f}%, +1={100*num_pos/total:.1f}%")
    print(f"Alpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
    print(f"Mu range: [{mu.min():.4f}, {mu.max():.4f}]")
    print()


def example_ssr():
    """Example: SSR column reordering."""
    print("=" * 50)
    print("Example 3: SSR Column Reordering")
    print("=" * 50)
    
    torch.manual_seed(42)
    W = torch.randn(256, 512)
    block_size = 128
    
    # Compute similarity matrix
    S = compute_cosine_similarity_matrix(W)
    print(f"Similarity matrix shape: {S.shape}")
    print(f"Similarity range: [{S.min():.4f}, {S.max():.4f}]")
    
    # Analyze block variance before reordering
    var_before = compute_block_variance(W, block_size)
    print(f"Block variances before SSR: {[f'{v:.4f}' for v in var_before]}")
    
    # Create SSR reorderer
    ssr = SSRReorderer(W, block_size=block_size, use_dynamic=False)
    W_reordered = ssr.reorder_weights(W)
    
    # Analyze block variance after reordering
    var_after = compute_block_variance(W_reordered, block_size)
    print(f"Block variances after SSR: {[f'{v:.4f}' for v in var_after]}")
    print(f"Average variance reduction: {100*(1-sum(var_after)/sum(var_before)):.1f}%")
    print()


def example_quantize_linear_layer():
    """Example: Quantize a linear layer."""
    print("=" * 50)
    print("Example 4: Quantize Linear Layer")
    print("=" * 50)
    
    torch.manual_seed(42)
    
    # Create a linear layer
    layer = nn.Linear(512, 256)
    W_original = layer.weight.data.clone()
    
    # Generate calibration data
    X = torch.randn(100, 32, 512)  # (batch, seq_len, features)
    
    # Quantize
    atq = AsymmetricTernaryQuantizer()
    X_flat = X.reshape(-1, 512)
    alpha, mu, T = atq.quantize(W_original, X_flat)
    
    # Compute quantized output
    W_quant = atq.dequantize(alpha, mu, T)
    
    # Compare outputs
    with torch.no_grad():
        out_original = layer(X)
        out_quant = X @ W_quant.T + layer.bias
    
    mse = ((out_original - out_quant) ** 2).mean().item()
    print(f"Output MSE: {mse:.6f}")
    
    # Memory analysis
    orig_bytes = W_original.numel() * 4  # float32
    quant_bytes = T.numel() + alpha.numel() * 2 + mu.numel() * 2  # int8 + float16
    print(f"Original size: {orig_bytes / 1024:.2f} KB")
    print(f"Quantized size: {quant_bytes / 1024:.2f} KB")
    print(f"Compression: {orig_bytes / quant_bytes:.2f}x")
    print()


def example_blockwise_quantization():
    """Example: Block-wise quantization as in PT2-LLM."""
    print("=" * 50)
    print("Example 5: Block-wise Quantization")
    print("=" * 50)
    
    torch.manual_seed(42)
    W = torch.randn(256, 512)
    X = torch.randn(128, 512)
    block_size = 128
    
    n, m = W.shape
    atq = AsymmetricTernaryQuantizer()
    
    # Storage for block parameters
    all_alpha = []
    all_mu = []
    T_full = torch.zeros(n, m, dtype=torch.int8)
    
    # Process blocks
    for start in range(0, m, block_size):
        end = min(start + block_size, m)
        W_block = W[:, start:end]
        X_block = X[:, start:end]
        
        alpha_b, mu_b, T_b = atq.quantize(W_block, X_block)
        
        all_alpha.append(alpha_b)
        all_mu.append(mu_b)
        T_full[:, start:end] = T_b.to(torch.int8)
    
    # Stack parameters
    alpha = torch.cat(all_alpha, dim=1)
    mu = torch.cat(all_mu, dim=1)
    
    print(f"Number of blocks: {len(all_alpha)}")
    print(f"Alpha shape: {alpha.shape}")
    print(f"Mu shape: {mu.shape}")
    print(f"T shape: {T_full.shape}")
    
    # Reconstruct and evaluate
    W_recon = torch.zeros_like(W)
    for b in range(len(all_alpha)):
        start = b * block_size
        end = min((b + 1) * block_size, m)
        W_recon[:, start:end] = alpha[:, b:b+1] * T_full[:, start:end].float() + mu[:, b:b+1]
    
    err = compute_quantization_error(W, W_recon)
    print(f"Total quantization error: {err:.4f}")
    print()


if __name__ == '__main__':
    example_atq_basic()
    example_atq_full()
    example_ssr()
    example_quantize_linear_layer()
    example_blockwise_quantization()
