"""
hdgt/models/__init__.py

Phase 2 Model Definitions: Qwen2.5-VL Embedder and Heterogeneous Graph Neural Network (HDGT).
"""

from .qwen_embedder import QwenVLEmbedder
from .hdgt_gnn import HDGTHeteroGNN

__all__ = ["QwenVLEmbedder", "HDGTHeteroGNN"]
