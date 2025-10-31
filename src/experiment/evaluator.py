"""
Comprehensive evaluation suite for LLMs.
"""

from typing import Dict, List, Optional, Any
from collections import defaultdict
import re

import numpy as np
import torch


class LLMEvaluator:
    def __init__(self, model: torch.nn.Module, tokenizer: Any, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

    def evaluate_batch(
        self,
        input_batch: torch.Tensor,
        target_batch: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, float]:
        input_batch = input_batch.to(self.device)
        target_batch = target_batch.to(self.device)

        with torch.no_grad():
            if attention_mask is not None:
                logits = self.model(input_batch, attention_mask=attention_mask.to(self.device))
            else:
                logits = self.model(input_batch)

            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target_batch.reshape(-1),
                ignore_index=-100,
                reduction="mean",
            )

            predictions = torch.argmax(logits, dim=-1)
            mask = target_batch != -100
            correct = (predictions == target_batch) & mask
            accuracy = correct.sum().item() / mask.sum().item() if mask.sum() > 0 else 0.0

            perplexity = torch.exp(loss).item()

        return {"loss": loss.item(), "perplexity": perplexity, "accuracy": accuracy}

    def evaluate_loader(self, data_loader, num_batches: Optional[int] = None, verbose: bool = True) -> Dict[str, float]:
        self.model.eval()
        metrics_accumulator = defaultdict(list)
        total = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                if num_batches is not None and i >= num_batches:
                    break
                if len(batch) == 2:
                    input_batch, target_batch = batch
                    attention_mask = None
                else:
                    input_batch, target_batch, attention_mask = batch
                batch_metrics = self.evaluate_batch(input_batch, target_batch, attention_mask)
                for k, v in batch_metrics.items():
                    metrics_accumulator[k].append(v)
                if verbose and (i + 1) % 10 == 0:
                    print(f"Evaluated {i + 1}/{total} batches...", end="\r")
        return {k: float(np.mean(v)) for k, v in metrics_accumulator.items()}

    # ---- Simple BLEU/ROUGE implementations for quick eval ----
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+|[^\w\s]", text.lower())

    def calculate_bleu(self, predictions: List[str], references: List[str], n: int = 4) -> float:
        from collections import Counter

        def ngrams(tokens: List[str], k: int) -> Counter:
            return Counter(tuple(tokens[i : i + k]) for i in range(len(tokens) - k + 1))

        scores = []
        for pred, ref in zip(predictions, references):
            pt = self._tokenize(pred)
            rt = self._tokenize(ref)
            precisions = []
            for k in range(1, n + 1):
                p = ngrams(pt, k)
                r = ngrams(rt, k)
                matches = sum((p & r).values())
                total = sum(p.values())
                precisions.append(matches / total if total > 0 else 0.0)
            gm = np.exp(np.mean(np.log(np.array(precisions) + 1e-10)))
            bp = min(1.0, np.exp(1 - len(rt) / (len(pt) + 1e-10)))
            scores.append(bp * gm)
        return float(np.mean(scores)) if scores else 0.0

    def calculate_rouge(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        def lcs_len(a: List[str], b: List[str]) -> int:
            m, n = len(a), len(b)
            dp = [[0] * (n + 1) for _ in range(m + 1)]
            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
            return dp[m][n]

        precisions, recalls, f1s = [], [], []
        for pred, ref in zip(predictions, references):
            pt = self._tokenize(pred)
            rt = self._tokenize(ref)
            lcs = lcs_len(pt, rt)
            p = lcs / len(pt) if pt else 0.0
            r = lcs / len(rt) if rt else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f1)
        return {"rouge-l_precision": float(np.mean(precisions)), "rouge-l_recall": float(np.mean(recalls)), "rouge-l_f1": float(np.mean(f1s))}






