import argparse
import torch
import gc
import os
from model import load_model_for_quantization
from utils import load_quantized_model, evaluate_perplexity, set_seed
def cleanup_memory():
    """清理显存以防 OOM"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(description="Evaluate Base and/or Quantized Model PPL")
    parser.add_argument('--model', type=str, required=True, help="Original Model ID (e.g. Qwen/Qwen3-0.5B)")
    parser.add_argument('--base', action='store_true', help="Evaluate base (FP16) model")
    parser.add_argument('--quant', type=str, default=None, help="Path to quantized model.pt file (if provided, evaluate quantized model)")
    parser.add_argument('--eval_dataset', type=str, default='c4', choices=['wikitext', 'c4', 'ptb'])
    parser.add_argument('--seq_len', type=int, default=512)
    parser.add_argument('--max_samples', type=int, default=None, help="Max number of samples for C4 dataset")
    parser.add_argument('--block_size', type=int, default=128, help="Block size used during quantization")
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Check if at least one evaluation mode is specified
    if not args.base and args.quant is None:
        parser.error("At least one of --base or --quant must be specified")
        return

    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"==================================================")
    print(f"🔍 Model Evaluation Task")
    print(f"   Model:           {args.model}")
    print(f"   Evaluate Base:   {args.base}")
    print(f"   Evaluate Quant:  {args.quant is not None}")
    if args.quant:
        print(f"   Quantized Path:  {args.quant}")
    print(f"   Dataset:         {args.eval_dataset}")
    print(f"==================================================")

    ppl_base = None
    ppl_quant = None

    # ---------------------------------------------------------
    # Step 1: Evaluate Base Model (FP16) if requested
    # ---------------------------------------------------------
    if args.base:
        print(f"\n1️⃣  Loading Base Model (FP16)...")
        model, tokenizer = load_model_for_quantization(args.model, device=args.device)
        
        print(f"   Evaluating Base Model PPL...")
        
        # Use dataset config based on dataset choice
        dataset_config = {
            'wikitext': 'wikitext-2-raw-v1',
            'c4': None,
            'ptb': None
        }.get(args.eval_dataset, 'wikitext-2-raw-v1')
        
        ppl_base = evaluate_perplexity(
            model, 
            tokenizer, 
            dataset_name=args.eval_dataset,
            dataset_config=dataset_config,
            seq_len=args.seq_len,
            device=device
        )
        print(f"   ✅ Base PPL: {ppl_base:.2f}")

        # Clean up to free VRAM for next step
        del model
        cleanup_memory()
        print("   (Memory cleaned)")

    # ---------------------------------------------------------
    # Step 2: Evaluate Quantized Model if requested
    # ---------------------------------------------------------
    if args.quant is not None:
        print(f"\n2️⃣  Loading Quantized Model...")
        
        # 清理内存
        cleanup_memory()
        
        # Load structure again
        print(f"   Loading model structure (CPU)...")
        model, tokenizer = load_model_for_quantization(args.model, device='cpu')  # Load to CPU first
        
        # Load quantized weights
        print(f"   Loading weights from {args.quant}...")
        if os.path.isdir(args.quant):
            checkpoint_file = os.path.join(args.quant, "model.pt")
        else:
            checkpoint_file = args.quant
            
        try:
            model, quant_params = load_quantized_model(model, checkpoint_file, block_size=args.block_size)
            
            # 清理内存
            cleanup_memory()
            
            # Verify weights were actually loaded
            if not quant_params:
                print("   ⚠️ Warning: No quantized parameters were loaded!")
            
            print(f"   Moving model to {device}...")
            model = model.to(device)
            
        except Exception as e:
            print(f"   ❌ Error loading quantized model: {e}")
            import traceback
            traceback.print_exc()
            return

        print(f"   Evaluating Quantized Model PPL...")
        
        dataset_config = {
            'wikitext': 'wikitext-2-raw-v1',
            'c4': None,
            'ptb': None
        }.get(args.eval_dataset, 'wikitext-2-raw-v1')
        
        ppl_quant = evaluate_perplexity(
            model, 
            tokenizer, 
            dataset_name=args.eval_dataset,
            dataset_config=dataset_config,
            seq_len=args.seq_len,
            device=device
        )
        print(f"   ✅ Quantized PPL: {ppl_quant:.2f}")
        
        del model
        cleanup_memory()
        print("   (Memory cleaned)")

    # ---------------------------------------------------------
    # Step 3: Summary
    # ---------------------------------------------------------
    print(f"\n==================================================")
    print(f"📊 Final Results ({args.eval_dataset})")
    print(f"==================================================")
    
    if ppl_base is not None:
        print(f"Base Model PPL:       {ppl_base:.2f}")
    
    if ppl_quant is not None:
        print(f"Quantized PPL:        {ppl_quant:.2f}")
    
    if ppl_base is not None and ppl_quant is not None:
        diff = ppl_quant - ppl_base
        print(f"PPL Degradation:      +{diff:.2f}")
        
        if diff < 1.0:
            print(f"Conclusion:           🚀 Excellent! (Negligible loss)")
        elif diff < 5.0:
            print(f"Conclusion:           ✅ Good. (Acceptable loss)")
        else:
            print(f"Conclusion:           ⚠️  High degradation. (Check algorithm)")
    
    print(f"==================================================")

if __name__ == '__main__':
    main()