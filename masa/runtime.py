import logging
import random
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


class AverageMeter:
    def __init__(self, name: str, fmt: str = ":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, value, count: int = 1) -> None:
        numeric = float(value)
        self.val = numeric
        self.sum += numeric * count
        self.count += count
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        template = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return template.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches: int, meters: Iterable[AverageMeter], prefix: str = ""):
        self.batch_format = self._batch_format(num_batches)
        self.meters = list(meters)
        self.prefix = prefix

    def display(self, batch: int) -> None:
        entries = [self.prefix + self.batch_format.format(batch)]
        entries.extend(str(meter) for meter in self.meters)
        print("\t".join(entries))

    @staticmethod
    def _batch_format(num_batches: int) -> str:
        digits = len(str(num_batches))
        field = "{:" + str(digits) + "d}"
        return "[" + field + "/" + field.format(num_batches) + "]"


@torch.no_grad()
def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: Sequence[int] = (1,),
) -> list[torch.Tensor]:
    maximum = max(topk)
    batch_size = target.size(0)
    prediction = output.topk(maximum, dim=1, largest=True, sorted=True).indices.t()
    correct = prediction.eq(target.view(1, -1).expand_as(prediction))
    return [
        correct[:k].reshape(-1).float().sum().mul_(100.0 / batch_size)
        for k in topk
    ]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_logger(output_dir: str | Path, log_name: str, debug: bool = False) -> logging.Logger:
    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"masa.{log_name}")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG if debug else logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)-8s: %(message)s")
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(output_path / log_name)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger

