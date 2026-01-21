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
    
    # Tokenize
    tokens = tokenizer(all_text, return_tensors='pt')['input_ids'][0]
    
    # Create samples
    samples = []
    for i in range(num_samples):
        start_idx = random.randint(0, len(tokens) - seq_len - 1)
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


def compute_compressed_size(quantized_params: Dict[str, Dict[str, torch.Tensor]]) -> Tuple[float, Dict[str, int]]:
    """
    Compute the real compressed size of quantized parameters.

    Args:
        quantized_params: Dictionary mapping layer names to quantized parameters

    Returns:
        Tuple of (total_size_gb, breakdown_dict)
    """
    total_bytes = 0
    breakdown = {'T': 0, 'alpha': 0, 'mu': 0, 'perm': 0}

    for name, params in quantized_params.items():
        # T: int8 (1 byte per weight)
        t_bytes = params['T'].numel() * 1
        breakdown['T'] += t_bytes
        total_bytes += t_bytes

        # alpha: FP16 (2 bytes per element)
        alpha_bytes = params['alpha'].numel() * 2
        breakdown['alpha'] += alpha_bytes
        total_bytes += alpha_bytes

        # mu: FP16 (2 bytes per element)
        mu_bytes = params['mu'].numel() * 2
        breakdown['mu'] += mu_bytes
        total_bytes += mu_bytes

        # perm: int32 (4 bytes per element)
        perm_bytes = params['perm'].numel() * 4
        breakdown['perm'] += perm_bytes
        total_bytes += perm_bytes

    total_gb = total_bytes / (1024 ** 3)
    return total_gb, breakdown


def compute_packed_size(quantized_params: Dict[str, Dict[str, torch.Tensor]]) -> float:
    """
    Compute the theoretical packed size using 2-bit ternary encoding.

    Ternary values {-1, 0, 1} can be packed as 2 bits each (4 values per byte).

    Args:
        quantized_params: Dictionary mapping layer names to quantized parameters

    Returns:
        Total size in GB with 2-bit packing
    """
    total_bytes = 0

    for name, params in quantized_params.items():
        # T: 2 bits per weight (packed 4 values per byte)
        t_bytes = (params['T'].numel() + 3) // 4  # ceil division
        total_bytes += t_bytes

        # alpha: FP16 (2 bytes)
        total_bytes += params['alpha'].numel() * 2

        # mu: FP16 (2 bytes)
        total_bytes += params['mu'].numel() * 2

        # perm: int32 (4 bytes) - could be optimized to int16 for smaller models
        total_bytes += params['perm'].numel() * 4

    return total_bytes / (1024 ** 3)


def save_quantized_model(model: nn.Module,
                          save_path: str,
                          quantized_params: Dict[str, Dict[str, torch.Tensor]],
                          save_compressed: bool = True):
    """
    Save quantized model parameters.

    Args:
        model: The quantized model (used to extract config)
        save_path: Path to save the checkpoint
        quantized_params: Dictionary mapping layer names to quantized parameters
        save_compressed: If True, save only compressed ternary params (smaller file)
                        If False, also save full model state dict (for compatibility)
    """
    # Prepare compressed parameters
    compressed_params = {}
    for layer_name, params in quantized_params.items():
        compressed_params[layer_name] = {
            'T': params['T'].to(torch.int8),         # 1 byte per weight
            'alpha': params['alpha'].half(),          # FP16
            'mu': params['mu'].half(),                # FP16
            'perm': params['perm'].to(torch.int32),   # Permutation indices
        }

    save_dict = {
        'quantized_params': compressed_params,
        'format_version': '2.0',  # New compressed format
    }

    # Try to save model config if available
    if hasattr(model, 'config'):
        try:
            save_dict['model_config'] = model.config.to_dict()
        except Exception:
            pass  # Config not serializable, skip

    # Optionally include full state dict for compatibility
    if not save_compressed:
        save_dict['model_state_dict'] = model.state_dict()

    torch.save(save_dict, save_path)

    # Report size
    compressed_size, _ = compute_compressed_size(quantized_params)
    print(f"Saved compressed model: {compressed_size:.3f} GB")


def load_quantized_model(model: nn.Module, 
                          load_path: str,
                          block_size: int = 128) -> Tuple[nn.Module, Dict]:
    """
    Load quantized model parameters with strict key mapping.
    
    Args:
        model: The model to load weights into
        load_path: Path to the quantized checkpoint
        block_size: Block size used during quantization (default 128)
        
    Returns:
        Tuple of (model, quantized_params)
    """
    print(f"   Loading checkpoint from {load_path}...")
    checkpoint = torch.load(load_path, map_location='cpu', weights_only=False)
    
    if 'quantized_params' not in checkpoint:
        # 兼容旧格式
        if 'model_state_dict' in checkpoint:
            print("   Loading from old format (model_state_dict)...")
            model.load_state_dict(checkpoint['model_state_dict'])
            return model, {}
        raise ValueError("Checkpoint format not recognized")

    quantized_params = checkpoint['quantized_params']
    print(f"   Found {len(quantized_params)} quantized layers")
    
    # 1. 自动推断模型的前缀 (Prefix Detection)
    model_keys = list(model.state_dict().keys())
    prefix = ""
    
    for key in model_keys:
        if "layers." in key and ".weight" in key:
            # Extract prefix before "layers.X."
            parts = key.split("layers.")
            if len(parts) >= 2:
                prefix = parts[0] + "layers"
                break
        elif ".h." in key:  # Bloom/GPT2 style
            parts = key.split(".h.")
            if len(parts) >= 2:
                prefix = parts[0] + ".h"
                break
    
    if not prefix:
        print("   ⚠️ Warning: Could not detect model prefix. Assuming 'model.layers'...")
        prefix = "model.layers"

    print(f"   Detected model prefix: '{prefix}'")
    
    # 2. 重构权重
    state_dict = model.state_dict()
    updated_count = 0
    failed_keys = []
    
    import gc
    
    with torch.no_grad():
        for idx, (layer_name, params) in enumerate(quantized_params.items()):
            # layer_name 格式: "layer_0.self_attn.q_proj"
            # 解析层号和子模块名
            try:
                parts = layer_name.split('.')
                layer_idx_str = parts[0].split('_')[1]  # "layer_0" -> "0"
                submodule_name = '.'.join(parts[1:])     # "self_attn.q_proj"
            except (IndexError, ValueError):
                print(f"   ⚠️ Skipping malformed layer name: {layer_name}")
                continue
            
            # 构造真实的 state_dict key
            # 目标: "model.layers.0.self_attn.q_proj.weight"
            real_key = f"{prefix}.{layer_idx_str}.{submodule_name}.weight"
            
            # 尝试多种命名方式
            possible_keys = [
                real_key,
                f"model.{prefix}.{layer_idx_str}.{submodule_name}.weight",  # model.model.layers...
                f"transformer.{prefix}.{layer_idx_str}.{submodule_name}.weight",  # GPT style
            ]
            
            found_key = None
            for key in possible_keys:
                if key in state_dict:
                    found_key = key
                    break
            
            if found_key is None:
                failed_keys.append((layer_name, real_key))
                if idx < 5:  # Only print first few failures
                    print(f"   ⚠️ Key not found: {layer_name} -> tried {real_key}")
                continue

            # 开始重构
            T = params['T'].float()
            alpha = params['alpha'].float()
            mu = params['mu'].float()
            perm = params.get('perm', None)
            
            out_features, in_features = T.shape
            num_blocks = alpha.shape[1] if alpha.dim() > 1 else 1
            W_reconstructed = torch.zeros_like(T)
            
            for b in range(num_blocks):
                start = b * block_size
                end = min((b + 1) * block_size, in_features)
                
                if alpha.dim() > 1:
                    alpha_b = alpha[:, b:b+1]
                    mu_b = mu[:, b:b+1]
                else:
                    alpha_b = alpha.unsqueeze(1)
                    mu_b = mu.unsqueeze(1)
                
                T_block = T[:, start:end]
                W_reconstructed[:, start:end] = alpha_b * T_block + mu_b
                
                # 立即释放块内临时张量
                del alpha_b, mu_b, T_block
            
            # Apply inverse permutation if needed
            if perm is not None:
                inv_perm = torch.argsort(perm)
                W_reconstructed = W_reconstructed[:, inv_perm]
                del inv_perm, perm
            
            # 立即释放参数张量
            del T, alpha, mu
            
            # 更新 State Dict
            state_dict[found_key] = W_reconstructed
            updated_count += 1
            
            # Debug: Print first layer reconstruction info
            if idx == 0:
                print(f"   DEBUG: First layer '{found_key}'")
                print(f"          Shape: {W_reconstructed.shape}")
                print(f"          Mean: {W_reconstructed.mean().item():.6f}")
                print(f"          Std: {W_reconstructed.std().item():.6f}")
            
            # 更频繁的垃圾回收和进度报告
            if (idx + 1) % 10 == 0:
                print(f"   Reconstructed {idx + 1}/{len(quantized_params)} layers...")
                gc.collect()
            
            # 释放重建后的张量引用
            del W_reconstructed

    print(f"\n   ✅ Successfully updated {updated_count}/{len(quantized_params)} layers")
    
    if failed_keys:
        print(f"   ⚠️ Failed to load {len(failed_keys)} layers:")
        for layer_name, attempted_key in failed_keys[:5]:
            print(f"      - {layer_name} (tried: {attempted_key})")
        if len(failed_keys) > 5:
            print(f"      ... and {len(failed_keys) - 5} more")
    
    # 3. 加载回模型
    model.load_state_dict(state_dict, strict=False)
    
    return model, quantized_params
           