from pathlib import Path
from typing import Sequence

import torch
from torchvision import datasets, transforms


COMMON_CORRUPTIONS = (
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
)

_NORMALIZE = transforms.Normalize(
    mean=(0.485, 0.456, 0.406),
    std=(0.229, 0.224, 0.225),
)
_CORRUPTION_TRANSFORM = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        _NORMALIZE,
    ]
)


class EvaluationImageFolder(datasets.ImageFolder):
    def __init__(self, root: str | Path):
        super().__init__(str(root), transform=_CORRUPTION_TRANSFORM)
        self._all_samples = tuple(self.samples)

    def set_specific_subset(self, indices: Sequence[int]) -> None:
        if not indices:
            raise ValueError("The label-shift index sequence is empty.")
        minimum = min(indices)
        maximum = max(indices)
        if minimum < 0 or maximum >= len(self._all_samples):
            raise IndexError(
                f"Label-shift indices must be within [0, {len(self._all_samples) - 1}], "
                f"got [{minimum}, {maximum}]."
            )
        self.samples = [self._all_samples[index] for index in indices]
        self.imgs = self.samples
        self.targets = [target for _, target in self.samples]


def build_dataset(
    data_corruption: str | Path,
    corruption: str,
    level: int,
) -> EvaluationImageFolder:
    if corruption not in COMMON_CORRUPTIONS:
        raise ValueError(f"Unsupported corruption: {corruption}")
    root = Path(data_corruption).expanduser().resolve() / corruption / str(level)
    if not root.is_dir():
        raise FileNotFoundError(f"Corruption directory not found: {root}")
    return EvaluationImageFolder(root)


def build_loader(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    shuffle: bool,
    workers: int,
    use_cuda: bool,
) -> torch.utils.data.DataLoader:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=use_cuda,
    )

