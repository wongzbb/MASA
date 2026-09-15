import argparse
import math
import re
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import timm
import torch
from timm.models.vision_transformer import _load_weights
from torch.utils.data import ConcatDataset, Dataset

from masa import MASA, SemanticProtoConfig, collect_params, configure_model
from masa.data import COMMON_CORRUPTIONS, build_dataset, build_loader
from masa.runtime import AverageMeter, ProgressMeter, accuracy, make_logger, set_seed


ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"

MODEL_CONFIGS = {
    "resnet50_gn_timm": {
        "architecture": "resnet50_gn",
        "covariance": "cov_resnet50_gn_timm.npy",
        "margin": 0.8,
        "margin_l0": 0.8,
        "reweight_threshold": 3.0,
    },
    "vitbase_timm": {
        "architecture": "vit_base_patch16_224",
        "covariance": "cov_vitbase_timm.npy",
        "margin": 1.0,
        "margin_l0": 1.0,
        "reweight_threshold": 1.5,
    },
}


class CorruptionTaggedDataset(Dataset):
    def __init__(self, dataset: Dataset, corruption: str):
        self.dataset = dataset
        self.corruption = corruption

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return image, target, self.corruption


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got: {value}")


def parse_topks(raw: str, num_classes: int) -> tuple[int, ...]:
    values = [int(token) for token in re.split(r"[,;\s]+", raw.strip()) if token]
    topks = tuple(dict.fromkeys(values))
    if not topks:
        raise ValueError("--report_topk must contain at least one integer.")
    if any(value < 1 or value > num_classes for value in topks):
        raise ValueError(
            f"--report_topk values must be within [1, {num_classes}], got {topks}."
        )
    return topks


def parse_corruptions(raw: str | None) -> list[str]:
    if not raw:
        return list(COMMON_CORRUPTIONS)
    values = [token for token in re.split(r"[,;\s]+", raw.strip()) if token]
    unsupported = sorted(set(values) - set(COMMON_CORRUPTIONS))
    if unsupported:
        raise ValueError(f"Unsupported corruptions: {unsupported}")
    return values


def make_accuracy_meters(topks: Sequence[int]) -> list[AverageMeter]:
    return [AverageMeter(f"Acc@{value}", ":6.2f") for value in topks]


def update_accuracy_meters(
    output: torch.Tensor,
    target: torch.Tensor,
    topks: Sequence[int],
    meters: Sequence[AverageMeter],
) -> None:
    for meter, value in zip(meters, accuracy(output, target, topk=topks)):
        meter.update(value, target.size(0))


def format_accuracy(meters: Sequence[AverageMeter], topks: Sequence[int]) -> str:
    return " and ".join(
        f"top{topk}: {meter.avg:.5f}" for topk, meter in zip(topks, meters)
    )


def update_per_corruption(
    output: torch.Tensor,
    target: torch.Tensor,
    tags: Sequence[str] | None,
    topks: Sequence[int],
    meters_by_corruption: dict[str, list[AverageMeter]],
) -> None:
    if tags is None:
        return
    tag_list = list(tags)
    for corruption, meters in meters_by_corruption.items():
        positions = [index for index, tag in enumerate(tag_list) if tag == corruption]
        if not positions:
            continue
        index_tensor = torch.as_tensor(positions, device=target.device)
        update_accuracy_meters(
            output.index_select(0, index_tensor),
            target.index_select(0, index_tensor),
            topks,
            meters,
        )


def choose_device(gpu: int) -> torch.device:
    if gpu < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    if gpu >= torch.cuda.device_count():
        raise ValueError(
            f"GPU index {gpu} is unavailable; detected {torch.cuda.device_count()} device(s)."
        )
    torch.cuda.set_device(gpu)
    return torch.device(f"cuda:{gpu}")


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        try:
            state = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            state = torch.load(path, map_location="cpu")

    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            nested = state.get(key)
            if isinstance(nested, dict):
                state = nested
                break
    if not isinstance(state, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")

    return {
        key.removeprefix("module."): value
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }


def configure_model_settings(args) -> None:
    config = MODEL_CONFIGS[args.model]
    log_classes = math.log(args.num_class)

    margin = config["margin"] if args.masa_margin is None else args.masa_margin
    margin_l0 = config["margin_l0"] if args.masa_margin_l0 is None else args.masa_margin_l0
    args.masa_margin = float(margin) * log_classes
    args.masa_margin_l0 = float(margin_l0) * log_classes

    if args.reweight_threshold is None:
        args.reweight_threshold = (
            5.0 if args.exp_type == "bs1" else config["reweight_threshold"]
        )

    if args.model == "resnet50_gn_timm":
        args.lr = (
            (0.00025 / 64) * args.test_batch_size * 2
            if args.test_batch_size < 32
            else 0.00025
        )
    else:
        args.lr = (0.001 / 64) * args.test_batch_size

    if args.exp_type == "bs1":
        args.lr *= 2
    args.lr *= args.weight_lr

    covariance_path = ASSET_DIR / config["covariance"]
    args.sigmas = torch.from_numpy(np.load(covariance_path, allow_pickle=False))


def load_classifier(args, logger) -> torch.nn.Module:
    checkpoint_path = Path(args.model_checkpoint).expanduser().resolve()
    config = MODEL_CONFIGS[args.model]
    model = timm.create_model(config["architecture"], pretrained=False)

    if checkpoint_path.suffix.lower() == ".npz":
        if args.model != "vitbase_timm":
            raise ValueError("NPZ checkpoints are supported only for vitbase_timm.")
        _load_weights(model, str(checkpoint_path))
    else:
        state_dict = load_state_dict(checkpoint_path)
        model.load_state_dict(state_dict, strict=True)

    logger.info("Loaded classifier checkpoint: %s", checkpoint_path)
    return model.to(args.device)


def load_label_shift_indices(path: str | Path) -> list[int]:
    index_path = Path(path).expanduser().resolve()
    values = np.load(index_path, allow_pickle=False).reshape(-1)
    if not np.issubdtype(values.dtype, np.integer):
        rounded = np.rint(values)
        if not np.allclose(values, rounded):
            raise ValueError(f"Label-shift indices must be integral: {index_path}")
        values = rounded
    return values.astype(np.int64).tolist()


def make_single_loader(args, corruption: str):
    dataset = build_dataset(args.data_corruption, corruption, args.level)
    shuffle = args.if_shuffle
    if args.exp_type == "label_shifts":
        dataset.set_specific_subset(args.label_shift_indices_values)
        shuffle = False
    return build_loader(
        dataset,
        batch_size=args.test_batch_size,
        shuffle=shuffle,
        workers=args.workers,
        use_cuda=args.device.type == "cuda",
    )


def make_mix_loader(args, corruptions: Sequence[str]):
    datasets = [
        CorruptionTaggedDataset(
            build_dataset(args.data_corruption, corruption, args.level),
            corruption,
        )
        for corruption in corruptions
    ]
    return build_loader(
        ConcatDataset(datasets),
        batch_size=args.test_batch_size,
        shuffle=args.if_shuffle,
        workers=args.workers,
        use_cuda=args.device.type == "cuda",
    )


def validate_paths(parser: argparse.ArgumentParser, args) -> None:
    data_path = Path(args.data_corruption).expanduser()
    checkpoint_path = Path(args.model_checkpoint).expanduser()
    if not data_path.is_dir():
        parser.error(f"--data_corruption is not a directory: {data_path}")
    if not checkpoint_path.is_file():
        parser.error(f"--model_checkpoint is not a file: {checkpoint_path}")

    if args.exp_type == "label_shifts":
        index_path = Path(args.label_shift_indices).expanduser()
        if not index_path.is_file():
            parser.error(f"--label_shift_indices is not a file: {index_path}")

    if not args.semantic_enabled:
        return

    mllm_type = args.semantic_mllm_type
    if mllm_type not in {"none", "hash", "off"}:
        if not args.semantic_mllm_model:
            parser.error(
                "--semantic_mllm_model is required when a real MLLM backend is enabled."
            )
        if not Path(args.semantic_mllm_model).expanduser().is_dir():
            parser.error(
                f"--semantic_mllm_model is not a directory: {args.semantic_mllm_model}"
            )

    if args.semantic_text_encoder_checkpoint:
        text_path = Path(args.semantic_text_encoder_checkpoint).expanduser()
        if not text_path.is_file():
            parser.error(
                "--semantic_text_encoder_checkpoint is not a file: "
                f"{args.semantic_text_encoder_checkpoint}"
            )
    elif not args.semantic_hash_fallback:
        parser.error(
            "Provide --semantic_text_encoder_checkpoint or enable "
            "--semantic_hash_fallback."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="MASA test-time adaptation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_corruption", required=True)
    parser.add_argument("--model_checkpoint", required=True)
    parser.add_argument("--output", default="./outputs")
    parser.add_argument(
        "--label_shift_indices",
        default=str(ASSET_DIR / "label_shift_indices.npy"),
    )

    parser.add_argument("--method", default="masa", choices=("masa",))
    parser.add_argument(
        "--exp_type",
        default="label_shifts",
        choices=("bs1", "mix_shifts", "label_shifts"),
    )
    parser.add_argument(
        "--model",
        default="resnet50_gn_timm",
        choices=tuple(MODEL_CONFIGS),
    )
    parser.add_argument("--seed", default=2024, type=int)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--workers", default=8, type=int)
    parser.add_argument("--test_batch_size", default=64, type=int)
    parser.add_argument("--if_shuffle", default=True, type=str2bool)
    parser.add_argument("--max_batches", default=0, type=int)
    parser.add_argument("--debug", default=False, type=str2bool)
    parser.add_argument("--level", default=5, type=int)
    parser.add_argument("--corruptions", default=None)
    parser.add_argument("--report_topk", default="1,5")
    parser.add_argument("--plpd_threshold", default=0.2, type=float)
    parser.add_argument("--weight_lr", default=1.0, type=float)
    parser.add_argument("--masa_margin", default=None, type=float)
    parser.add_argument("--masa_margin_l0", default=None, type=float)
    parser.add_argument("--weight_tau", default=1.2, type=float)
    parser.add_argument("--weight_reg", default=0.5, type=float)
    parser.add_argument("--reweight_threshold", default=None, type=float)

    parser.add_argument("--semantic_enabled", default=True, type=str2bool)
    parser.add_argument(
        "--semantic_update_mode",
        default="norm_affine",
        choices=("norm_affine", "memory_only"),
    )
    parser.add_argument(
        "--semantic_mllm_type",
        default="none",
        choices=("none", "hash", "off", "qwen", "llava", "blip2", "blip"),
    )
    parser.add_argument("--semantic_mllm_model", default=None)
    parser.add_argument("--semantic_text_encoder_checkpoint", default=None)
    parser.add_argument("--semantic_hash_fallback", default=True, type=str2bool)
    parser.add_argument("--semantic_max_clusters", default=64, type=int)
    parser.add_argument("--semantic_window_size", default=128, type=int)
    parser.add_argument("--semantic_refresh_interval", default=128, type=int)
    parser.add_argument("--semantic_anchor_batch", default=1, type=int)
    parser.add_argument("--semantic_anchor_neighbors", default=4, type=int)
    parser.add_argument("--semantic_anchor_temperature", default=0.07, type=float)
    parser.add_argument("--semantic_tau_assign", default=0.7, type=float)
    parser.add_argument("--semantic_tau_q", default=0.1, type=float)
    parser.add_argument("--semantic_tau_beta", default=0.1, type=float)
    parser.add_argument("--semantic_tau_rel_store", default=0.7, type=float)
    parser.add_argument("--semantic_tau_ri_anchor", default=10.0, type=float)
    parser.add_argument("--semantic_tau_margin", default=0.05, type=float)
    parser.add_argument("--semantic_tau_flip", default=0.5, type=float)
    parser.add_argument("--semantic_drift_threshold", default=0.45, type=float)
    parser.add_argument("--semantic_coverage_threshold", default=0.25, type=float)
    parser.add_argument("--semantic_confirm_min_count", default=2, type=int)
    parser.add_argument("--semantic_beta_proto", default=0.1, type=float)
    parser.add_argument("--semantic_gamma_f", default=1.0, type=float)
    parser.add_argument("--semantic_gamma_p", default=1.0, type=float)
    parser.add_argument("--semantic_gamma_s", default=0.25, type=float)
    parser.add_argument("--semantic_alpha_v", default=0.45, type=float)
    parser.add_argument("--semantic_alpha_p", default=0.25, type=float)
    parser.add_argument("--semantic_alpha_s", default=0.20, type=float)
    parser.add_argument("--semantic_alpha_b", default=0.10, type=float)
    parser.add_argument("--semantic_alpha_age", default=0.02, type=float)
    parser.add_argument("--semantic_eta0", default=0.25, type=float)
    parser.add_argument("--semantic_rho_q", default=0.05, type=float)
    parser.add_argument("--semantic_mllm_batch_size", default=4, type=int)
    parser.add_argument("--semantic_prompt", default=None)
    return parser


def run_adaptation(args, loader, logger, topks, corruption_names):
    classifier = load_classifier(args, logger)
    classifier = configure_model(classifier)
    parameters, parameter_names = collect_params(classifier)
    if not parameters:
        raise RuntimeError("No normalization affine parameters were selected.")
    logger.info("Adapted parameters: %s", parameter_names)

    optimizer = torch.optim.SGD(parameters, args.lr, momentum=0.9)
    semantic_config = SemanticProtoConfig.from_args(args)
    model = MASA(
        classifier,
        optimizer,
        margin=args.masa_margin,
        margin_L0=args.masa_margin_l0,
        weight_reg=args.weight_reg,
        reweight_threshold=args.reweight_threshold,
        sigmas=args.sigmas,
        batch_size=args.test_batch_size,
        weight_tau=args.weight_tau,
        semantic_cfg=semantic_config,
        plpd_threshold=args.plpd_threshold,
    )
    set_seed(args.seed)

    meters = make_accuracy_meters(topks)
    per_corruption = {
        name: make_accuracy_meters(topks) for name in corruption_names
    }
    progress = ProgressMeter(len(loader), meters, prefix="Test: ")
    start = time.time()

    for batch_index, batch in enumerate(loader):
        images = batch[0].to(args.device, non_blocking=True)
        targets = batch[1].to(args.device, non_blocking=True)
        output = model(images)

        update_accuracy_meters(output, targets, topks, meters)
        tags = batch[2] if len(batch) > 2 else None
        update_per_corruption(
            output,
            targets,
            tags,
            topks,
            per_corruption,
        )

        if batch_index % args.print_freq == 0:
            progress.display(batch_index)
        if args.max_batches > 0 and batch_index + 1 >= args.max_batches:
            break

    logger.info("Elapsed seconds: %.2f", time.time() - start)
    return meters, per_corruption


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.test_batch_size < 1:
        parser.error("--test_batch_size must be positive.")
    if args.workers < 0:
        parser.error("--workers cannot be negative.")
    if args.max_batches < 0:
        parser.error("--max_batches cannot be negative.")
    if args.level < 1 or args.level > 5:
        parser.error("--level must be within [1, 5].")

    validate_paths(parser, args)
    args.num_class = 1000
    args.device = choose_device(args.gpu)
    if args.device.type == "cuda":
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if args.exp_type == "bs1":
        args.test_batch_size = 1
    configure_model_settings(args)

    topks = parse_topks(args.report_topk, args.num_class)
    corruptions = parse_corruptions(args.corruptions)
    args.print_freq = max(1, 50000 // 20 // args.test_batch_size)

    if args.exp_type == "label_shifts":
        args.label_shift_indices_values = load_label_shift_indices(
            args.label_shift_indices
        )
    else:
        args.label_shift_indices_values = None

    output_dir = Path(args.output).expanduser().resolve()
    log_name = (
        time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        + f"-masa-{args.model}-level{args.level}-seed{args.seed}.txt"
    )
    logger = make_logger(output_dir, log_name, args.debug)
    logger.info(
        "Experiment: type=%s model=%s device=%s corruptions=%s",
        args.exp_type,
        args.model,
        args.device,
        corruptions,
    )
    logger.info(
        "Semantic memory: enabled=%s mode=%s mllm=%s",
        args.semantic_enabled,
        args.semantic_update_mode,
        args.semantic_mllm_type,
    )
    if args.exp_type == "label_shifts":
        logger.info(
            "Label-shift ordering: indices=%s",
            Path(args.label_shift_indices).expanduser().resolve(),
        )

    history: list[list[float]] = []
    if args.exp_type == "mix_shifts":
        set_seed(args.seed)
        loader = make_mix_loader(args, corruptions)
        set_seed(args.seed)
        meters, per_corruption = run_adaptation(
            args,
            loader,
            logger,
            topks,
            corruptions,
        )
        logger.info("Result under mix_shifts: %s", format_accuracy(meters, topks))
        for name in corruptions:
            item_meters = per_corruption[name]
            logger.info(
                "Result under mix_shifts/%s: %s (count: %d)",
                name,
                format_accuracy(item_meters, topks),
                item_meters[0].count,
            )
        history.append([meter.avg for meter in meters])
    else:
        for corruption in corruptions:
            set_seed(args.seed)
            loader = make_single_loader(args, corruption)
            set_seed(args.seed)
            meters, _ = run_adaptation(args, loader, logger, topks, ())
            logger.info(
                "Result under %s: %s",
                corruption,
                format_accuracy(meters, topks),
            )
            history.append([meter.avg for meter in meters])

    mean_values = np.asarray(history, dtype=np.float64).mean(axis=0)
    logger.info(
        "Mean result: %s",
        " and ".join(
            f"top{topk}: {value:.5f}"
            for topk, value in zip(topks, mean_values)
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

