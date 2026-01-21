import os
from huggingface_hub import snapshot_download

# 设置环境变量指向镜像站
# 2026年1月 SOTA 模型列表
models_to_download = [
    # --- Group 1: Gemma 3 (4B) ---
    # Google 2025年3月发布，原生多模态，架构非常新
    #"google/gemma-3-4b-it",
    
    # --- Group 3: Qwen 3 (14B) ---
    # 阿里 2025年9月发布，如果你的显存有 24G 可以跑这个，非常强
    # 如果显存不够，可以注释掉这两行
    "Qwen/Qwen3-0.6B",
    #"Qwen/Qwen3-4B-Instruct",
]

print(f"=== 开始下载 2026 SOTA 模型组 ({len(models_to_download)} 个) ===")

for model_id in models_to_download:
    print(f"\n[正在下载]: {model_id}")
    try:
        # 只下载必要文件，排除 GGUF 等重复格式
        snapshot_download(
            repo_id=model_id,
            allow_patterns=["*.json", "*.model", "*.safetensors", "*.bin", "*.py", "*.txt"],
            resume_download=True,
            # Ministral 等模型可能需要 token，请确保终端已登录
            token=True 
        )
        print(f"√ {model_id} 下载完成！")
    except Exception as e:
        print(f"× {model_id} 下载失败: {e}")
        print("  提示: Ministral 和 Gemma 3 可能需要去 HuggingFace 网页签署协议。")

print("\n=== 所有下载任务结束 ===")