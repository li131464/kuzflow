#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DeepSeek-R1-Distill-Qwen-1.5B 模型下载脚本
"""

import os
from pathlib import Path
from huggingface_hub import snapshot_download
import torch

def download_deepseek_model():
    """下载DeepSeek-R1-Distill-Qwen-1.5B模型"""
    
    # 设置模型保存路径
    current_dir = Path(__file__).parent
    model_dir = current_dir / "DeepSeek-R1-Distill-Qwen-1.5B"
    
    print(f"模型将下载到: {model_dir}")
    
    # 检查GPU可用性
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"检测到设备: {device}")
    
    try:
        print("开始下载DeepSeek-R1-Distill-Qwen-1.5B模型...")
        print("这可能需要几分钟时间，请耐心等待...")
        
        # 下载模型
        model_path = snapshot_download(
            repo_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            local_dir=str(model_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        
        print(f"模型下载完成！保存路径: {model_path}")
        
        # 验证模型文件
        required_files = [
            "config.json",
            "tokenizer.json", 
            "tokenizer_config.json",
            "generation_config.json"
        ]
        
        print("\n验证模型文件:")
        for file in required_files:
            file_path = model_dir / file
            if file_path.exists():
                print(f"✓ {file}")
            else:
                print(f"✗ {file} (可能不是必需的)")
        
        # 检查模型权重文件
        weight_files = list(model_dir.glob("*.safetensors")) + list(model_dir.glob("*.bin"))
        if weight_files:
            print(f"✓ 找到 {len(weight_files)} 个模型权重文件")
        else:
            print("✗ 未找到模型权重文件")
        
        return True
        
    except Exception as e:
        print(f"下载模型时出错: {e}")
        return False

if __name__ == "__main__":
    success = download_deepseek_model()
    if success:
        print("\n模型下载成功！现在可以使用 deepseek.py 进行推理了。")
    else:
        print("\n模型下载失败，请检查网络连接或重试。") 