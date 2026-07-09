from dataclasses import dataclass

import torch


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += float(value) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> tuple[int, int]:
    preds = logits.argmax(dim=1)
    correct = (preds == targets).sum().item()
    return int(correct), int(targets.numel())
