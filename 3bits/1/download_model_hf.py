import os
from huggingface_hub import snapshot_download

def download_qwen3_models(local_dir_base="./qwen3_models"):
    """
    自动下载 Qwen3-0.5B 基础版和指令微调版
    """
    # 定义模型 ID
    # 注意：在官方仓库中，该尺寸可能被标记为 Qwen/Qwen3-0.5B 或 Qwen/Qwen3-0.6B
    # 这里以您要求的 0.5B 为准，请根据实际 Repo ID 微调
    models = {
        "base": "Qwen/Qwen3-1.7B",
        "instruct": "Qwen/Qwen3-0.5B-Instruct"
    }

    for model_type, repo_id in models.items():
        print(f"正在开始下载 {model_type} 模型: {repo_id}...")
        
        # 构造本地保存路径
        local_dir = os.path.join(local_dir_base, repo_id.split('/')[-1])
        
        try:
            # 执行下载
            # resume_download=True 支持断点续传
            # max_workers 可提高下载并发数
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=local_dir,
                local_dir_use_symlinks=False,  # 直接保存文件而非链接
                resume_download=True,
                token=None  # 如果是私有模型，请在此处填写 HF Token
            )
            print(f"✅ {model_type} 下载成功！保存在: {path}\n")
        except Exception as e:
            print(f"❌ 下载 {repo_id} 时出错: {e}")

if __name__ == "__main__":
    download_qwen3_models()