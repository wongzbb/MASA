# MASA Anonymous Review Package

This directory contains the minimal source needed to evaluate MASA.

## Package layout

```text
MASA/
|-- main.py
|-- masa/
|   |-- __init__.py
|   |-- adaptation.py
|   |-- data.py
|   |-- runtime.py
|   `-- semantic_memory.py
|-- assets/
|   |-- cov_resnet50_gn_timm.npy
|   |-- cov_vitbase_timm.npy
|   `-- label_shift_indices.npy
|-- requirements.txt
`-- README.md
```

## Environment

Pinned dependency versions are listed in `requirements.txt`.

```bash
python -m pip install -r requirements.txt
python main.py --help
```

## Required local assets

Datasets and model weights are intentionally not bundled. Supply these paths on
the command line:

- `--data_corruption`: ImageNet-C root with
  `<corruption>/<level>/<class>/<image>` layout.
- `--model_checkpoint`: classifier checkpoint in `.safetensors`, `.pth`,
  or `.npz` format.
- `--semantic_mllm_model`: local MLLM directory for a real MLLM run.
- `--semantic_text_encoder_checkpoint`: local text-encoder checkpoint.


## Evaluation commands

Run commands from this directory. The following command evaluates label shift
with a local Qwen-family backend:

```bash
python main.py \
  --method masa \
  --exp_type label_shifts \
  --model resnet50_gn_timm \
  --data_corruption /path/to/imagenet-c \
  --model_checkpoint /path/to/classifier.safetensors \
  --semantic_mllm_type qwen \
  --semantic_mllm_model /path/to/qwen-model \
  --semantic_text_encoder_checkpoint /path/to/text-encoder.pt \
  --semantic_hash_fallback false \
  --output ./outputs/qwen_label_shift
```

Mixed corruption shift:

```bash
python main.py \
  --method masa \
  --exp_type mix_shifts \
  --model resnet50_gn_timm \
  --data_corruption /path/to/imagenet-c \
  --model_checkpoint /path/to/classifier.safetensors \
  --semantic_mllm_type qwen \
  --semantic_mllm_model /path/to/qwen-model \
  --semantic_text_encoder_checkpoint /path/to/text-encoder.pt \
  --semantic_hash_fallback false \
  --report_topk 1,2 \
  --output ./outputs/qwen_mix_shift
```

Single-sample adaptation with a BLIP-family backend:

```bash
python main.py \
  --method masa \
  --exp_type bs1 \
  --model resnet50_gn_timm \
  --data_corruption /path/to/imagenet-c \
  --model_checkpoint /path/to/classifier.safetensors \
  --semantic_mllm_type blip2 \
  --semantic_mllm_model /path/to/blip-model \
  --semantic_text_encoder_checkpoint /path/to/text-encoder.pt \
  --semantic_hash_fallback false \
  --output ./outputs/blip_bs1
```

`--max_batches 0` runs the complete evaluation. The bundled label-shift index
file is the default ordering for the reported label-shift setting; another
ordering can be supplied with `--label_shift_indices`.

