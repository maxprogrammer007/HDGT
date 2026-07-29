"""
hdgt/models/qwen_embedder.py

Phase 2 — Multimodal Vision-Language Feature Extraction using Qwen2.5-VL.
"""

from typing import List, Union, Optional
import torch
import torch.nn as nn
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration


class QwenVLEmbedder(nn.Module):
    """
    Qwen2.5-VL feature extractor for encoding text nodes, layout headers,
    and questions into multimodal embedding vectors.
    """

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        device: str = "cuda:0" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.model_id = model_id
        self.device = torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()
        
        # Get hidden dimension size from model config
        self.hidden_dim = getattr(self.model.config, "hidden_size", 2048)

    @torch.no_grad()
    def embed_texts(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: int = 128,
    ) -> torch.Tensor:
        """
        Extract dense text embeddings for a list of string tokens/lines.
        
        Parameters
        ----------
        texts : List[str]
            List of text strings.
        batch_size : int
            Batch size for parallel forward pass.
        max_length : int
            Maximum token length per text sequence.
            
        Returns
        -------
        torch.Tensor of shape [N, hidden_dim] in float32.
        """
        if not texts:
            return torch.zeros((0, self.hidden_dim), dtype=torch.float32)

        embeddings_list = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            # Replace empty strings with a single space
            batch_texts = [t.strip() if t.strip() else " " for t in batch_texts]
            
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length,
            )
            
            # Move to model's first device
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            outputs = self.model.model(**inputs, output_hidden_states=True)
            # Use mean pooling over token sequence (last hidden state)
            last_hidden = outputs.last_hidden_state  # [B, T, D]
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            sum_embeddings = torch.sum(last_hidden * mask, dim=1)
            sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)
            mean_pooled = (sum_embeddings / sum_mask).to(torch.float32).cpu()
            
            embeddings_list.append(mean_pooled)

        return torch.cat(embeddings_list, dim=0)

    @torch.no_grad()
    def embed_query(self, query: str) -> torch.Tensor:
        """Embed a single question query string."""
        return self.embed_texts([query], batch_size=1)
