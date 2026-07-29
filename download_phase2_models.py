"""
download_phase2_models.py

Phase 2 Preparation: Download Qwen2.5-VL-3B-Instruct model weights and processor from HuggingFace Hub.
"""

import sys
import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

def download_model(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct"):
    print("=" * 60)
    print(f"  Downloading Phase 2 Vision-Language Model: {model_id}")
    print("=" * 60)
    
    print("[1/2] Loading Processor & Tokenizer...")
    processor = AutoProcessor.from_pretrained(model_id)
    print("Processor loaded successfully!")

    print("[2/2] Downloading Model Weights (PyTorch / Safetensors)...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print(f"Model successfully loaded on devices: {model.hf_device_map}")
    print("=" * 60)
    print("  Phase 2 Model Download & Verification Complete!")
    print("=" * 60)

if __name__ == "__main__":
    download_model()
