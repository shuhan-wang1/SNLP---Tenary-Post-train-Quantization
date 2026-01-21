# PT²-LLM: Post-Training Ternarization for Large Language Models

Implementation of the PT²-LLM paper, a post-training quantization framework that compresses LLMs to 1.58 bits per weight using ternary values {-1, 0, +1}.

## Method Overview

### Asymmetric Ternary Quantizer (ATQ)

The core quantizer with two-stage refinement:

1. **Iterative Ternary Fitting (ITF)**: Alternates between optimal grid construction and flexible rounding to minimize quantization error `E_w = ||W - αT - μ||²_F`

2. **Activation-aware Grid Alignment (AGA)**: Refines the ternary grid to minimize output error `E_x = ||WX - (αT + μ)X||²_F`

### Structural Similarity-based Reordering (SSR)

Reorders weight columns based on cosine similarity to:
- Create more homogeneous blocks for ternarization
- Group outliers together to reduce their distorting effect

### Key Equations

**Asymmetric Initialization** (Eq. 4-5):
```
μ = (1/m) Σ W_{:,j}           # Row-wise mean offset
W_f = W - μ                    # Centered weights
Δ ≈ 0.75/m Σ|W_f_{:,j}|       # Threshold
α = Σ(T_{:,j} · W_f_{:,j}) / Σ|T_{:,j}|
```

**Optimal Grid Construction** (Eq. 9):
```
α* = (m · (W ◦ T)1 - (T1) ◦ (W1)) / (m · (T ◦ T)1 - (T1)²)
μ* = ((T ◦ T)1 ◦ (W1) - (T1) ◦ [(W ◦ T)1]) / (m · (T ◦ T)1 - (T1)²)
```

**Flexible Rounding** (Eq. 10):
```
Z_ij = (W_ij - μ*_i) / α*_i
T*_ij = argmin_{t∈{-1,0,1}} |Z_ij - t|
```

**SSR Similarity** (Eq. 15-16):
```
S_ij = (W_{:,i}^T W_{:,j}) / (||W_{:,i}||₂ ||W_{:,j}||₂)
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Quantization

```bash
python main.py --model meta-llama/Llama-2-7b-hf --output ./quantized
```

### With Evaluation

```bash
python main.py --model meta-llama/Llama-2-7b-hf --output ./quantized --eval
```

### Options

```
--model         HuggingFace model name or path (required)
--output        Output directory (default: ./quantized_model)
--block_size    Block size for quantization (default: 128)
--num_samples   Number of calibration samples (default: 128)
--seq_len       Sequence length for calibration (default: 2048)
--no_ssr        Disable SSR reordering
--percdamp      GPTQ dampening factor (default: 0.01)
--eval          Evaluate perplexity after quantization
--eval_dataset  Dataset for evaluation (default: wikitext)
--seed          Random seed (default: 42)
--device        Device to use (default: cuda)
```

### Python API

```python
from pt2_llm import PT2LLMQuantizer, load_model_for_quantization

# Load model
model, tokenizer = load_model_for_quantization("meta-llama/Llama-2-7b-hf")

# Create quantizer
quantizer = PT2LLMQuantizer(
    model=model,
    tokenizer=tokenizer,
    model_type='llama2',
    block_size=128,
    num_calibration_samples=128,
    use_ssr=True
)

# Quantize
quantized_params = quantizer.quantize()
```

### Using Individual Components

```python
from pt2_llm import AsymmetricTernaryQuantizer

# Quantize a weight matrix
atq = AsymmetricTernaryQuantizer()
alpha, mu, T = atq.quantize(W, X)  # W: weights, X: activations

# Dequantize
W_approx = atq.dequantize(alpha, mu, T)
```

## File Structure

```
pt2_llm/
├── __init__.py      # Package exports
├── quantizer.py     # ATQ with ITF and AGA
├── reorder.py       # SSR column reordering
├── gptq.py          # GPTQ framework integration
├── model.py         # Model handling and TernaryLinear layer
├── utils.py         # Data loading, evaluation utilities
├── main.py          # Entry point and PT2LLMQuantizer
└── requirements.txt # Dependencies
```

## Expected Results

Based on the paper, PT²-LLM achieves:

| Model | Bits | WikiText2 PPL | Avg. Accuracy |
|-------|------|---------------|---------------|
| LLaMA-7B | 1.58 | 11.39 | 45.07% |
| LLaMA-13B | 1.58 | 9.11 | 48.64% |
| LLaMA-65B | 1.58 | 6.62 | 55.95% |
| LLaMA-2-7B | 1.58 | 11.56 | 43.33% |
| LLaMA-2-70B | 1.58 | 6.27 | 55.87% |

## Reference

```bibtex
@article{yan2025pt2llm,
  title={PT2-LLM: Post-Training Ternarization for Large Language Models},
  author={Yan, Xianglong and Bao, Chengzhu and Li, Zhiteng and Zhang, Tianao and Yang, Kaicheng and Qin, Haotong and Xie, Ruobing and Sun, Xingwu and Zhang, Yulun},
  journal={arXiv preprint arXiv:2510.03267},
  year={2025}
}
```
