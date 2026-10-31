"""Optional cross-encoder reranker (MedCPT Cross-Encoder by default)."""
from __future__ import annotations

import numpy as np


class CrossEncoderReranker:
    def __init__(self, model: str, device: str | None = None, batch_size: int = 16):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        from src.index.dense import _device

        self.device = _device(device)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForSequenceClassification.from_pretrained(model).to(self.device).eval()
        self.batch_size = batch_size

    def score(self, query: str, docs: list[str]) -> np.ndarray:
        import torch

        out = []
        with torch.no_grad():
            for i in range(0, len(docs), self.batch_size):
                pairs = [[query, d] for d in docs[i:i + self.batch_size]]
                enc = self.tok(pairs, truncation=True, padding=True, return_tensors="pt",
                               max_length=512).to(self.device)
                out.append(self.model(**enc).logits.squeeze(-1).float().cpu().numpy().reshape(-1))
        return np.concatenate(out) if out else np.zeros(0)
