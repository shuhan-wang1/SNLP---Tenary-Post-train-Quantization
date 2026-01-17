"""
PT2-LLM: Post-Training Ternarization for Large Language Models

A post-training quantization framework that compresses LLMs to 1.58 bits per weight
using ternary quantization {-1, 0, +1}.

Components:
- AsymmetricTernaryQuantizer: Core quantizer with ITF and AGA
- SSRReorderer: Structural Similarity-based Reordering
- GPTQ: Block-wise quantization with error compensation
- PT2LLMQuantizer: High-level quantization interface
"""

from quantizer import (
    AsymmetricTernaryQuantizer,
    compute_quantization_error,
    compute_output_error
)

from reorder import (
    SSRReorderer,
    compute_cosine_similarity_matrix,
    select_next_block_ssr,
    apply_permutation
)

from gptq import GPTQ, GPTQQuantizer

from model import (
    TernaryLinear,
    load_model_for_quantization,
    get_llm_layers,
    get_model_type
)

from utils import (
    get_calibration_data,
    evaluate_perplexity,
    compute_bits_per_weight,
    pack_ternary,
    unpack_ternary
)

from main import PT2LLMQuantizer

__version__ = '1.0.0'
__all__ = [
    'AsymmetricTernaryQuantizer',
    'SSRReorderer', 
    'GPTQ',
    'GPTQQuantizer',
    'TernaryLinear',
    'PT2LLMQuantizer',
    'load_model_for_quantization',
    'get_calibration_data',
    'evaluate_perplexity',
]
