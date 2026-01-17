"""
PT2-LLM: Model Handling and Layer Wrappers

Provides utilities for:
1. Model loading and preparation
2. Layer-by-layer quantization
3. Ternary layer implementation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import gc


class TernaryLinear(nn.Module):
    """
    Ternary linear layer storing quantized parameters.
    
    Stores:
    - T: Ternary matrix {-1, 0, +1} packed as int8
    - alpha: Row-wise scaling factors (one per block)
    - mu: Row-wise offsets (one per block)
    - perm: Column permutation indices
    """
    
    def __init__(self, 
                 in_features: int, 
                 out_features: int,
                 block_size: int = 128,
                 bias: bool = True,
                 dtype: torch.dtype = torch.float16):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.block_size = block_size
        
        num_blocks = (in_features + block_size - 1) // block_size
        
        # Ternary matrix stored as int8 (values: -1, 0, 1)
        self.register_buffer('T', torch.zeros(out_features, in_features, dtype=torch.int8))
        
        # Per-block scaling and offset
        self.register_buffer('alpha', torch.ones(out_features, num_blocks, dtype=dtype))
        self.register_buffer('mu', torch.zeros(out_features, num_blocks, dtype=dtype))
        
        # Column permutation
        self.register_buffer('perm', torch.arange(in_features, dtype=torch.long))
        self.register_buffer('inv_perm', torch.arange(in_features, dtype=torch.long))
        
        # Bias
        if bias:
            self.register_buffer('bias', torch.zeros(out_features, dtype=dtype))
        else:
            self.bias = None
    
    def set_quantized_params(self, 
                             alpha: torch.Tensor, 
                             mu: torch.Tensor, 
                             T: torch.Tensor,
                             perm: torch.Tensor,
                             bias: Optional[torch.Tensor] = None):
        """Set quantized parameters."""
        self.T.copy_(T.to(torch.int8))
        self.alpha.copy_(alpha)
        self.mu.copy_(mu)
        self.perm.copy_(perm)
        self.inv_perm.copy_(torch.argsort(perm))
        
        if bias is not None and self.bias is not None:
            self.bias.copy_(bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with dequantization.
        
        1. Permute input features
        2. Dequantize weights block-by-block
        3. Perform matmul
        """
        # Apply permutation to input
        x_perm = x[..., self.perm]
        
        # Dequantize weight
        W = self._dequantize()
        
        # Linear operation
        out = torch.nn.functional.linear(x_perm, W[:, self.inv_perm])
        
        if self.bias is not None:
            out = out + self.bias
        
        return out
    
    def _dequantize(self) -> torch.Tensor:
        """Dequantize ternary weights to floating point."""
        W = torch.zeros(self.out_features, self.in_features, 
                       device=self.T.device, dtype=self.alpha.dtype)
        
        for b in range(self.alpha.shape[1]):
            start = b * self.block_size
            end = min((b + 1) * self.block_size, self.in_features)
            
            # W_block = alpha_b * T_block + mu_b
            T_block = self.T[:, start:end].to(self.alpha.dtype)
            W[:, start:end] = self.alpha[:, b:b+1] * T_block + self.mu[:, b:b+1]
        
        return W
    
    def memory_footprint(self) -> int:
        """Compute memory footprint in bytes."""
        # T: int8 (1 byte per element)
        T_bytes = self.T.numel()
        
        # alpha, mu: float16 (2 bytes per element)
        alpha_bytes = self.alpha.numel() * 2
        mu_bytes = self.mu.numel() * 2
        
        # perm: int64 (8 bytes per element)
        perm_bytes = self.perm.numel() * 8
        
        # bias if present
        bias_bytes = self.bias.numel() * 2 if self.bias is not None else 0
        
        return T_bytes + alpha_bytes + mu_bytes + perm_bytes + bias_bytes


def get_model_layers(model: nn.Module) -> Dict[str, nn.Linear]:
    """Extract all linear layers from model."""
    layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            layers[name] = module
    return layers


def get_llm_layers(model: nn.Module, model_type: str = 'llama') -> List[nn.Module]:
    """Get transformer layers from different model architectures."""
    if model_type == 'llama' or model_type == 'llama2' or model_type == 'llama3':
        return model.model.layers
    elif model_type == 'qwen' or model_type == 'qwen3':
        return model.model.layers
    elif model_type == 'opt':
        return model.model.decoder.layers
    elif model_type == 'bloom':
        return model.transformer.h
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def find_linear_layers(module: nn.Module, prefix: str = '') -> Dict[str, nn.Linear]:
    """Recursively find all linear layers in a module."""
    linear_layers = {}
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        if isinstance(child, nn.Linear):
            linear_layers[full_name] = child
        else:
            linear_layers.update(find_linear_layers(child, full_name))
    return linear_layers


def replace_linear_with_ternary(model: nn.Module, 
                                 quantized_params: Dict[str, Dict[str, torch.Tensor]],
                                 block_size: int = 128) -> nn.Module:
    """
    Replace linear layers with ternary versions.
    
    Args:
        model: Original model
        quantized_params: Dict mapping layer names to quantized parameters
        block_size: Block size used during quantization
        
    Returns:
        Model with ternary layers
    """
    for name, params in quantized_params.items():
        # Navigate to parent module
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        
        # Get original layer
        layer_name = parts[-1]
        orig_layer = getattr(parent, layer_name)
        
        # Create ternary layer
        ternary_layer = TernaryLinear(
            in_features=orig_layer.in_features,
            out_features=orig_layer.out_features,
            block_size=block_size,
            bias=orig_layer.bias is not None,
            dtype=params['alpha'].dtype
        )
        
        # Set parameters
        ternary_layer.set_quantized_params(
            alpha=params['alpha'],
            mu=params['mu'],
            T=params['T'],
            perm=params['perm'],
            bias=orig_layer.bias.data if orig_layer.bias is not None else None
        )
        
        # Replace layer
        setattr(parent, layer_name, ternary_layer)
        
        # Free original weights
        del orig_layer
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return model


def load_model_for_quantization(model_name: str, 
                                 device: str = 'auto',
                                 dtype: torch.dtype = torch.float16) -> Tuple[nn.Module, Any]:
    """
    Load model for quantization.
    
    Args:
        model_name: HuggingFace model name or path
        device: Device to load model on
        dtype: Data type for model weights
        
    Returns:
        model: Loaded model
        tokenizer: Associated tokenizer
    """
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        use_fast=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=config,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )
    
    model.eval()
    
    return model, tokenizer


def get_model_type(model_name: str) -> str:
    """Infer model type from name."""
    model_name_lower = model_name.lower()
    
    if 'llama-3' in model_name_lower or 'llama3' in model_name_lower:
        return 'llama3'
    elif 'llama-2' in model_name_lower or 'llama2' in model_name_lower:
        return 'llama2'
    elif 'llama' in model_name_lower:
        return 'llama'
    elif 'qwen3' in model_name_lower:
        return 'qwen3'
    elif 'qwen' in model_name_lower:
        return 'qwen'
    elif 'opt' in model_name_lower:
        return 'opt'
    elif 'bloom' in model_name_lower:
        return 'bloom'
    else:
        return 'llama'  # Default assumption


def compute_model_size(model: nn.Module, quantized: bool = False) -> float:
    """Compute model size in GB."""
    total_bytes = 0
    
    for name, param in model.named_parameters():
        total_bytes += param.numel() * param.element_size()
    
    for name, buffer in model.named_buffers():
        total_bytes += buffer.numel() * buffer.element_size()
    
    return total_bytes / (1024 ** 3)


def compute_compression_ratio(original_size: float, quantized_size: float) -> float:
    """Compute compression ratio."""
    return original_size / quantized_size
