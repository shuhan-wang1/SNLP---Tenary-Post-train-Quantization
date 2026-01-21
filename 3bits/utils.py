"""
PT2-LLM: Utility Functions

Data loading, calibration, and evaluation utilities.
"""

import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any
from datasets import load_dataset
import random
import numpy as np


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_calibration_data(tokenizer, 
                         dataset_name: str = 'wikitext',
                         dataset_config: str = 'wikitext-2-raw-v1',
                         num_samples: int = 128,
                         seq_len: int = 2048,
                         seed: int = 42) -> List[torch.Tensor]:
    """
    Load calibration data for quantization.
    
    Args:
        tokenizer: HuggingFace tokenizer
        dataset_name: Dataset name
        dataset_config: Dataset configuration
        num_samples: Number of calibration samples
        seq_len: Sequence length per sample
        seed: Random seed
        
    Returns:
        List of tokenized calibration samples
    """
    set_seed(seed)
    
    if dataset_name == 'wikitext':
        dataset = load_dataset(dataset_name, dataset_config, split='train')
        text_column = 'text'
    elif dataset_name == 'c4':
        dataset = load_dataset('allenai/c4', 'en', split='train', streaming=True)
        dataset = dataset.take(num_samples * 10)
        text_column = 'text'
    elif dataset_name == 'ptb':
        dataset = load_dataset('ptb_text_only', 'penn_treebank', split='train')
        text_column = 'sentence'
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    # Concatenate all text
    if dataset_name == 'c4':
        all_text = '\n\n'.join([item[text_column] for item in dataset])
    else:
        all_text = '\n\n'.join(dataset[text_column])
    
    # [FIXED] Handle tokenization of large datasets
    # Temporarily disable max length check for the huge concatenated string
    original_max_len = tokenizer.model_max_length
    tokenizer.model_max_length = int(1e30) # Prevent warnings
    
    tokens = tokenizer(all_text, return_tensors='pt', truncation=False)['input_ids'][0]
    
    # Restore max length
    tokenizer.model_max_length = original_max_len
    
    # Create samples
    samples = []
    for i in range(num_samples):
        # Ensure we don't go out of bounds
        max_start = len(tokens) - seq_len - 1
        if max_start <= 0:
            raise ValueError("Dataset is too small for the requested sequence length")
            
        start_idx = random.randint(0, max_start)
        sample = tokens[start_idx:start_idx + seq_len].unsqueeze(0)
        samples.append(sample)
    
    return samples


def prepare_calibration_inputs(model: nn.Module,
                                samples: List[torch.Tensor],
                                device: torch.device) -> Dict[str, List[torch.Tensor]]:
    """
    Prepare calibration inputs by running forward pass and collecting activations.
    
    Args:
        model: The model to calibrate
        samples: List of input token tensors
        device: Device to run on
        
    Returns:
        Dictionary mapping layer names to collected activations
    """
    activations = {}
    hooks = []
    
    def make_hook(name):
        def hook(module, inp, out):
            if name not in activations:
                activations[name] = []
            # Store input to linear layer
            if isinstance(inp, tuple):
                inp = inp[0]
            activations[name].append(inp.detach().cpu())
        return hook
    
    # Register hooks on linear layers
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            hooks.append(module.register_forward_hook(make_hook(name)))
    
    # Run calibration
    model.eval()
    with torch.no_grad():
        for sample in samples:
            sample = sample.to(device)
            model(sample)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Concatenate activations
    for name in activations:
        activations[name] = torch.cat(activations[name], dim=0)
    
    return activations


@torch.no_grad()
def evaluate_perplexity(model: nn.Module,
                        tokenizer,
                        dataset_name: str = 'wikitext',
                        dataset_config: str = 'wikitext-2-raw-v1',
                        seq_len: int = 2048,
                        device: torch.device = None) -> float:
    """
    Evaluate perplexity on a dataset.
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer
        dataset_name: Dataset name
        dataset_config: Dataset config
        seq_len: Sequence length for evaluation
        device: Device to run on
        
    Returns:
        Perplexity score
    """
    if device is None:
        device = next(model.parameters()).device
    
    if dataset_name == 'wikitext':
        dataset = load_dataset(dataset_name, dataset_config, split='test')
        text = '\n\n'.join(dataset['text'])
    elif dataset_name == 'c4':
        dataset = load_dataset('allenai/c4', 'en', split='validation', streaming=True)
        dataset = list(dataset.take(1000))
        text = '\n\n'.join([item['text'] for item in dataset])
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    encodings = tokenizer(text, return_tensors='pt')
    input_ids = encodings['input_ids'].to(device)
    
    nlls = []
    prev_end_loc = 0
    seq_len = min(seq_len, input_ids.size(1))
    
    for begin_loc in range(0, input_ids.size(1), seq_len):
        end_loc = min(begin_loc + seq_len, input_ids.size(1))
        trg_len = end_loc - prev_end_loc
        
        input_chunk = input_ids[:, begin_loc:end_loc]
        target_ids = input_chunk.clone()
        target_ids[:, :-trg_len] = -100
        
        outputs = model(input_chunk, labels=target_ids)
        neg_log_likelihood = outputs.loss * trg_len
        nlls.append(neg_log_likelihood)
        
        prev_end_loc = end_loc
        if end_loc >= input_ids.size(1):
            break
    
    ppl = torch.exp(torch.stack(nlls).sum() / prev_end_loc)
    return ppl.item()


def pack_ternary(T: torch.Tensor) -> torch.Tensor:
    """
    Pack ternary values into compact representation.
    
    Each element in {-1, 0, 1} can be stored in 2 bits.
    Pack 4 ternary values into 1 byte.
    
    Args:
        T: Ternary tensor with values in {-1, 0, 1}
        
    Returns:
        Packed tensor
    """
    # Map {-1, 0, 1} to {0, 1, 2}
    T_mapped = (T + 1).to(torch.uint8)
    
    # Pad to multiple of 4
    orig_shape = T_mapped.shape
    flat = T_mapped.flatten()
    
    pad_len = (4 - len(flat) % 4) % 4
    if pad_len > 0:
        flat = torch.cat([flat, torch.zeros(pad_len, dtype=torch.uint8, device=T.device)])
    
    # Reshape for packing
    flat = flat.reshape(-1, 4)
    
    # Pack 4 values into 1 byte
    packed = (flat[:, 0] | (flat[:, 1] << 2) | (flat[:, 2] << 4) | (flat[:, 3] << 6))
    
    return packed, orig_shape


def unpack_ternary(packed: torch.Tensor, orig_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Unpack ternary values from compact representation.
    
    Args:
        packed: Packed tensor
        orig_shape: Original tensor shape
        
    Returns:
        Unpacked ternary tensor
    """
    # Unpack
    unpacked = torch.zeros(len(packed) * 4, dtype=torch.int8, device=packed.device)
    unpacked[0::4] = (packed & 0x03)
    unpacked[1::4] = ((packed >> 2) & 0x03)
    unpacked[2::4] = ((packed >> 4) & 0x03)
    unpacked[3::4] = ((packed >> 6) & 0x03)
    
    # Map back {0, 1, 2} to {-1, 0, 1}
    unpacked = unpacked - 1
    
    # Reshape
    total_elements = 1
    for dim in orig_shape:
        total_elements *= dim
    
    return unpacked[:total_elements].reshape(orig_shape)


def compute_bits_per_weight(model: nn.Module, include_scales: bool = True) -> float:
    """
    Compute average bits per weight for quantized model.
    
    For ternarization:
    - Ternary values: log2(3) ≈ 1.58 bits per weight
    - Plus overhead from scales (α, μ) per block
    
    Args:
        model: Quantized model
        include_scales: Whether to include scale overhead
        
    Returns:
        Average bits per weight
    """
    total_params = 0
    total_bits = 0
    
    for name, module in model.named_modules():
        if hasattr(module, 'T'):  # TernaryLinear
            n_weights = module.T.numel()
            total_params += n_weights
            
            # Ternary values: 1.58 bits each
            total_bits += n_weights * 1.58
            
            if include_scales:
                # Alpha and mu: 16 bits each, per block
                n_scales = module.alpha.numel() + module.mu.numel()
                total_bits += n_scales * 16
    
    if total_params == 0:
        return 16.0  # Full precision
    
    return total_bits / total_params


def save_quantized_model(model: nn.Module, 
                          save_path: str,
                          quantized_params: Dict[str, Dict[str, torch.Tensor]]):
    """Save quantized model parameters."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'quantized_params': quantized_params
    }
    torch.save(save_dict, save_path)


def load_quantized_model(model: nn.Module, 
                          load_path: str) -> Tuple[nn.Module, Dict]:
    """Load quantized model parameters."""
    checkpoint = torch.load(load_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint.get('quantized_params', {})