import hashlib
import json
import math
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import to_pil_image


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    return x / x.norm(dim=dim, keepdim=True).clamp_min(eps)


def normalize_prob(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x.clamp_min(eps) / x.clamp_min(eps).sum(dim=-1, keepdim=True)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    image = image.detach().cpu()
    if image.ndim == 4:
        image = image[0]
    std = IMAGENET_STD.to(image)
    mean = IMAGENET_MEAN.to(image)
    image = (image * std + mean).clamp(0.0, 1.0)
    return to_pil_image(image)


def parse_json_object(text: str) -> Dict[str, object]:
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return {"object_family": text.strip()[:80], "confidence": 0.4}
    try:
        data = json.loads(match.group(0))
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {"object_family": text.strip()[:80], "confidence": 0.4}


def sanitize_semantics(data: Dict[str, object], pred_hint: Optional[str] = None) -> Dict[str, object]:
    result = {
        "object_family": str(data.get("object_family") or pred_hint or "unknown object"),
        "scene": str(data.get("scene") or "unknown scene"),
        "style_shift": data.get("style_shift") or [],
        "viewpoint": str(data.get("viewpoint") or "unknown viewpoint"),
        "occlusion": str(data.get("occlusion") or "unknown"),
        "label_preserving_prob": float(data.get("label_preserving_prob", 0.5) or 0.5),
        "confidence": float(data.get("confidence", 0.5) or 0.5),
    }
    if isinstance(result["style_shift"], str):
        result["style_shift"] = [result["style_shift"]]
    if not isinstance(result["style_shift"], list):
        result["style_shift"] = list(result["style_shift"])
    result["confidence"] = max(0.0, min(1.0, result["confidence"]))
    result["label_preserving_prob"] = max(0.0, min(1.0, result["label_preserving_prob"]))
    return result


def semantics_to_phrases(data: Dict[str, object]) -> Tuple[List[str], List[float], Set[str], float]:
    slot_weights = {
        "object_family": 1.25,
        "style_shift": 1.10,
        "scene": 0.70,
        "viewpoint": 0.60,
        "occlusion": 0.55,
    }
    confidence = float(data.get("confidence", 0.5))
    label_preserving = float(data.get("label_preserving_prob", 0.5))
    phrases: List[str] = []
    weights: List[float] = []
    codes: Set[str] = set()

    for slot in ("object_family", "scene", "viewpoint", "occlusion"):
        value = str(data.get(slot) or "").strip().lower()
        if not value or value.startswith("unknown"):
            continue
        phrases.append(f"{slot}: {value}")
        weights.append(slot_weights[slot] * max(confidence, 0.1))
        codes.add(f"{slot}={value}")

    for value in data.get("style_shift", []) or []:
        value = str(value).strip().lower()
        if not value or value.startswith("unknown"):
            continue
        phrases.append(f"style_shift: {value}")
        weights.append(slot_weights["style_shift"] * max(confidence, 0.1))
        codes.add(f"style_shift={value}")

    if not phrases:
        phrases = ["object_family: unknown object"]
        weights = [0.1]
    return phrases, weights, codes, confidence * label_preserving


class HashTextEncoder:
    def __init__(self, dim: int = 512, device: str = "cpu"):
        self.dim = dim
        self.device = torch.device(device)

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        rows = []
        for text in texts:
            vec = torch.zeros(self.dim, dtype=torch.float32)
            for token in re.findall(r"[a-z0-9_]+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] % 2 == 0 else -1.0
                vec[idx] += sign
            if vec.norm() == 0:
                vec[0] = 1.0
            rows.append(safe_normalize(vec, dim=0))
        return torch.stack(rows, dim=0).to(self.device)


class SemanticTextEncoder:
    def __init__(self, checkpoint: Optional[str] = None, device: Optional[str] = None,
                 allow_hash_fallback: bool = False):
        self.checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint else None
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.allow_hash_fallback = allow_hash_fallback
        self.model = None
        self.tokenizer = None
        self.fallback = HashTextEncoder(device=str(self.device))
        self._loaded = False

    def _ensure_loaded(self):
        if self._loaded:
            return
        self._loaded = True

        if self.checkpoint is None:
            if self.allow_hash_fallback:
                return
            raise FileNotFoundError(
                "A local semantic text-encoder checkpoint is required. "
                "Pass --semantic_text_encoder_checkpoint."
            )
        if not self.checkpoint.is_file():
            raise FileNotFoundError(
                f"Semantic text-encoder checkpoint not found: {self.checkpoint}"
            )

        try:
            import clip  # type: ignore

            self.model, _ = clip.load(str(self.checkpoint), device=str(self.device), jit=False)
            self.model.eval()
            self.tokenizer = clip.tokenize
        except Exception:
            if not self.allow_hash_fallback:
                raise
            self.model = None
            self.tokenizer = None

    @torch.no_grad()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        self._ensure_loaded()
        if self.model is None or self.tokenizer is None:
            return self.fallback.encode(texts)
        tokens = self.tokenizer(list(texts), truncate=True).to(self.device)
        feats = self.model.encode_text(tokens).float()
        return safe_normalize(feats, dim=-1)


class MLLMAdapter:
    PROMPT = (
        "Describe the image with strict JSON only. Use this schema: "
        "{\"object_family\":\"...\",\"scene\":\"...\",\"style_shift\":[\"...\"],"
        "\"viewpoint\":\"...\",\"occlusion\":\"low|mid|high\","
        "\"label_preserving_prob\":0.0,\"confidence\":0.0}. "
        "Keep values short and do not add text outside JSON."
    )

    def __init__(self, model_type: str = "none", device: Optional[str] = None,
                 prompt: Optional[str] = None, batch_size: int = 4,
                 model_id: Optional[str] = None):
        self.model_type = (model_type or "none").lower()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.prompt = prompt or self.PROMPT
        self.batch_size = batch_size
        self.model_id = model_id
        self.processor = None
        self.model = None
        self._loaded = False

    def _model_source(self) -> str:
        if not self.model_id:
            raise FileNotFoundError(
                "A local MLLM directory is required. Pass --semantic_mllm_model."
            )
        model_dir = Path(self.model_id).expanduser().resolve()
        if not model_dir.is_dir():
            raise FileNotFoundError(f"MLLM directory not found: {model_dir}")
        return str(model_dir)

    def _ensure_loaded(self):
        if self._loaded or self.model_type in ("none", "hash", "off"):
            self._loaded = True
            return
        self._loaded = True
        model_dir = self._model_source()
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        from transformers import AutoProcessor  # type: ignore

        if self.model_type == "qwen":
            from transformers import Qwen2VLForConditionalGeneration  # type: ignore

            min_pixels = 256 * 28 * 28
            max_pixels = 1280 * 28 * 28
            self.processor = AutoProcessor.from_pretrained(
                model_dir,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                local_files_only=True,
            )
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map={"": str(self.device)},
                local_files_only=True,
            )
        elif self.model_type == "llava":
            from transformers import LlavaForConditionalGeneration  # type: ignore

            self.processor = AutoProcessor.from_pretrained(
                model_dir, local_files_only=True
            )
            self.model = LlavaForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map={"": str(self.device)},
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
        elif self.model_type in ("blip2", "blip"):
            from transformers import Blip2ForConditionalGeneration  # type: ignore

            self.processor = AutoProcessor.from_pretrained(
                model_dir, use_fast=True, local_files_only=True
            )
            self.model = Blip2ForConditionalGeneration.from_pretrained(
                model_dir,
                torch_dtype=dtype,
                device_map={"": str(self.device)},
                local_files_only=True,
            )
        else:
            raise ValueError(f"Unsupported MLLM type: {self.model_type}")

    @torch.no_grad()
    def query_batch(self, images: Sequence[Image.Image],
                    pred_hints: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
        pred_hints = pred_hints or [None] * len(images)
        if self.model_type in ("none", "hash", "off"):
            return [
                sanitize_semantics({
                    "object_family": hint or "unknown object",
                    "scene": "unknown scene",
                    "style_shift": [],
                    "viewpoint": "unknown viewpoint",
                    "occlusion": "unknown",
                    "label_preserving_prob": 0.5,
                    "confidence": 0.35,
                }, hint)
                for hint in pred_hints
            ]

        self._ensure_loaded()
        outputs: List[Dict[str, object]] = []
        for start in range(0, len(images), self.batch_size):
            batch_images = list(images[start:start + self.batch_size])
            batch_hints = list(pred_hints[start:start + self.batch_size])
            texts = self._generate(batch_images)
            for text, hint in zip(texts, batch_hints):
                outputs.append(sanitize_semantics(parse_json_object(text), hint))
        return outputs

    def _generate(self, images: Sequence[Image.Image]) -> List[str]:
        assert self.processor is not None and self.model is not None
        device = self.device
        model_type = self.model_type
        model = self.model
        processor = self.processor
        input_dtype = torch.float16 if device.type == "cuda" else torch.float32

        if model_type == "qwen":
            template = processor.tokenizer.apply_chat_template(
                [{"role": "user", "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            if isinstance(template, list):
                template = template[0]
            inputs = processor(
                text=[template] * len(images),
                images=list(images),
                padding=True,
                return_tensors="pt",
            ).to(device)
            generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
            generated = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs.input_ids, generated)
            ]
            return processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

        if model_type == "llava":
            results = []
            for image in images:
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                }]
                prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(device, input_dtype)
                generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
                text = processor.decode(generated[0], skip_special_tokens=True)
                results.append(text.split("assistant\n")[-1].split("ASSISTANT:")[-1].strip())
            return results

        prompts = [f"Question: {self.prompt} Answer:" for _ in images]
        inputs = processor(images=list(images), text=prompts, return_tensors="pt", padding=True).to(device, input_dtype)
        generated = model.generate(**inputs, max_new_tokens=160, do_sample=False)
        input_len = inputs.input_ids.shape[1]
        generated = generated[:, input_len:]
        return processor.batch_decode(generated, skip_special_tokens=True)


class SemanticCodec:
    def __init__(self, text_encoder: SemanticTextEncoder):
        self.text_encoder = text_encoder

    @torch.no_grad()
    def encode_one(self, semantics: Dict[str, object]) -> Tuple[torch.Tensor, Set[str], float]:
        phrases, weights, codes, confidence = semantics_to_phrases(semantics)
        feats = self.text_encoder.encode(phrases)
        weight = torch.tensor(weights, dtype=feats.dtype, device=feats.device)
        weight = weight / weight.sum().clamp_min(1e-12)
        feat = safe_normalize((feats * weight[:, None]).sum(dim=0), dim=0)
        return feat.detach().cpu(), codes, float(confidence)


@dataclass
class Anchor:
    feature: torch.Tensor
    sem: torch.Tensor
    code: Set[str]
    confidence: float
    age: int = 0


@dataclass
class Cluster:
    mu: torch.Tensor
    p_bar: torch.Tensor
    s_sem: Optional[torch.Tensor]
    b_code: Set[str]
    q_rel: float
    n: float = 1.0
    age: int = 0
    use: int = 1
    var: float = 0.0
    state: str = "candidate"
    score_sum: float = 0.0


@dataclass
class BufferedSample:
    image: torch.Tensor
    feature: torch.Tensor
    prob: torch.Tensor
    l_re: float
    l_ri: float
    margin: float
    flip: float
    reliability: float
    pred_hint: str


@dataclass
class SemanticProtoConfig:
    enabled: bool = True
    update_mode: str = "norm_affine"
    mllm_type: str = "none"
    text_encoder_checkpoint: Optional[str] = None
    allow_hash_fallback: bool = False
    max_clusters: int = 64
    window_size: int = 128
    refresh_interval: int = 64
    anchor_batch: int = 2
    anchor_neighbors: int = 4
    anchor_temperature: float = 0.07
    tau_assign: float = 0.6
    tau_q: float = 0.1
    tau_beta: float = 0.1
    tau_rel_store: float = 0.5
    tau_re_anchor: float = 1.0
    tau_ri_anchor: float = 10.0
    tau_margin: float = 0.05
    tau_flip: float = 0.5
    drift_threshold: float = 0.45
    coverage_threshold: float = 0.25
    confirm_min_count: int = 2
    beta_proto: float = 0.2
    gamma_f: float = 1.0
    gamma_p: float = 1.0
    gamma_s: float = 0.25
    alpha_v: float = 0.45
    alpha_p: float = 0.25
    alpha_s: float = 0.20
    alpha_b: float = 0.10
    alpha_age: float = 0.02
    eta0: float = 0.25
    rho_q: float = 0.05
    mllm_batch_size: int = 4
    mllm_model: Optional[str] = None
    prompt: Optional[str] = None

    @classmethod
    def from_args(cls, args) -> "SemanticProtoConfig":
        num_class = max(float(getattr(args, "num_class", 1000)), 2.0)
        return cls(
            enabled=bool(getattr(args, "semantic_enabled", True)),
            update_mode=getattr(args, "semantic_update_mode", "norm_affine"),
            mllm_type=getattr(args, "semantic_mllm_type", "none"),
            text_encoder_checkpoint=getattr(args, "semantic_text_encoder_checkpoint", None),
            allow_hash_fallback=bool(getattr(args, "semantic_hash_fallback", False)),
            max_clusters=int(getattr(args, "semantic_max_clusters", 64)),
            window_size=int(getattr(args, "semantic_window_size", 128)),
            refresh_interval=int(getattr(args, "semantic_refresh_interval", 128)),
            anchor_batch=int(getattr(args, "semantic_anchor_batch", 1)),
            anchor_neighbors=int(getattr(args, "semantic_anchor_neighbors", 4)),
            anchor_temperature=float(getattr(args, "semantic_anchor_temperature", 0.07)),
            tau_assign=float(getattr(args, "semantic_tau_assign", 0.7)),
            tau_q=float(getattr(args, "semantic_tau_q", 0.1)),
            tau_beta=float(getattr(args, "semantic_tau_beta", 0.1)),
            tau_rel_store=float(getattr(args, "semantic_tau_rel_store", 0.7)),
            tau_re_anchor=float(getattr(args, "masa_margin", 0.8 * math.log(num_class))),
            tau_ri_anchor=float(getattr(args, "semantic_tau_ri_anchor", math.log(num_class))),
            tau_margin=float(getattr(args, "semantic_tau_margin", 0.05)),
            tau_flip=float(getattr(args, "semantic_tau_flip", 0.5)),
            drift_threshold=float(getattr(args, "semantic_drift_threshold", 0.45)),
            coverage_threshold=float(getattr(args, "semantic_coverage_threshold", 0.25)),
            confirm_min_count=int(getattr(args, "semantic_confirm_min_count", 2)),
            beta_proto=float(getattr(args, "semantic_beta_proto", 0.1)),
            gamma_f=float(getattr(args, "semantic_gamma_f", 1.0)),
            gamma_p=float(getattr(args, "semantic_gamma_p", 1.0)),
            gamma_s=float(getattr(args, "semantic_gamma_s", 0.25)),
            alpha_v=float(getattr(args, "semantic_alpha_v", 0.45)),
            alpha_p=float(getattr(args, "semantic_alpha_p", 0.25)),
            alpha_s=float(getattr(args, "semantic_alpha_s", 0.20)),
            alpha_b=float(getattr(args, "semantic_alpha_b", 0.10)),
            alpha_age=float(getattr(args, "semantic_alpha_age", 0.02)),
            eta0=float(getattr(args, "semantic_eta0", 0.25)),
            rho_q=float(getattr(args, "semantic_rho_q", 0.05)),
            mllm_batch_size=int(getattr(args, "semantic_mllm_batch_size", 4)),
            mllm_model=getattr(args, "semantic_mllm_model", None),
            prompt=getattr(args, "semantic_prompt", None),
        )


class SemanticPrototypeMemory:
    def __init__(self, cfg: SemanticProtoConfig, device: Optional[str] = None):
        self.cfg = cfg
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.window: deque[BufferedSample] = deque(maxlen=cfg.window_size)
        self.coverage: deque[float] = deque(maxlen=cfg.window_size)
        self.anchors: deque[Anchor] = deque(maxlen=max(cfg.max_clusters * 4, cfg.anchor_neighbors))
        self.clusters: List[Cluster] = []
        self.step = 0
        self.pred_history: deque[int] = deque(maxlen=32)
        self.mllm = MLLMAdapter(cfg.mllm_type, device=str(self.device),
                                prompt=cfg.prompt, batch_size=cfg.mllm_batch_size,
                                model_id=cfg.mllm_model)
        self.codec = SemanticCodec(
            SemanticTextEncoder(
                cfg.text_encoder_checkpoint, device=str(self.device),
                allow_hash_fallback=cfg.allow_hash_fallback
            )
        )

    def reset(self):
        self.window.clear()
        self.coverage.clear()
        self.anchors.clear()
        self.clusters.clear()
        self.step = 0
        self.pred_history.clear()

    def recent_flip_rate(self, pred: int) -> float:
        if len(self.pred_history) < 2:
            self.pred_history.append(pred)
            return 0.0
        values = list(self.pred_history)
        flips = sum(int(values[i] != values[i - 1]) for i in range(1, len(values)))
        self.pred_history.append(pred)
        return flips / max(len(values) - 1, 1)

    def reliability(self, l_re: float, l_ri: float, margin: float, flip: float,
                    sem_conf: float = 1.0) -> float:
        gates = [
            l_re < self.cfg.tau_re_anchor,
            l_ri < self.cfg.tau_ri_anchor,
            margin > self.cfg.tau_margin,
            flip < self.cfg.tau_flip,
        ]
        return float(all(gates)) * float(sem_conf)

    def push_window(self, images: torch.Tensor, features: torch.Tensor, probs: torch.Tensor,
                    l_re: torch.Tensor, l_ri: torch.Tensor) -> List[float]:
        features = safe_normalize(features.detach().float(), dim=-1).cpu()
        probs = normalize_prob(probs.detach().float()).cpu()
        images_cpu = images.detach().cpu()
        l_re_cpu = l_re.detach().float().cpu()
        l_ri_cpu = l_ri.detach().float().cpu()
        top2 = probs.topk(k=min(2, probs.shape[1]), dim=1).values
        margins = top2[:, 0] - (top2[:, 1] if top2.shape[1] > 1 else 0.0)
        preds = probs.argmax(dim=1)
        reliabilities: List[float] = []
        for i in range(images_cpu.shape[0]):
            flip = self.recent_flip_rate(int(preds[i]))
            rel = self.reliability(float(l_re_cpu[i]), float(l_ri_cpu[i]),
                                   float(margins[i]), flip)
            reliabilities.append(rel)
            self.window.append(BufferedSample(
                image=images_cpu[i],
                feature=features[i],
                prob=probs[i],
                l_re=float(l_re_cpu[i]),
                l_ri=float(l_ri_cpu[i]),
                margin=float(margins[i]),
                flip=flip,
                reliability=rel,
                pred_hint=f"predicted_class_{int(preds[i])}",
            ))
        return reliabilities

    def maybe_refresh_anchors(self):
        if not self.cfg.enabled:
            return
        trigger = len(self.anchors) == 0
        trigger = trigger or (self.cfg.refresh_interval > 0 and self.step % self.cfg.refresh_interval == 0)
        trigger = trigger or self._window_drift() > self.cfg.drift_threshold
        if len(self.coverage) == self.coverage.maxlen:
            trigger = trigger or (sum(self.coverage) / len(self.coverage) < self.cfg.coverage_threshold)
        if not trigger:
            return
        samples = self._select_diverse_anchors()
        if not samples:
            return
        images = [tensor_to_pil(sample.image) for sample in samples]
        hints = [sample.pred_hint for sample in samples]
        sems = self.mllm.query_batch(images, hints)
        for sample, sem in zip(samples, sems):
            sem_vec, code, sem_conf = self.codec.encode_one(sem)
            conf = max(float(sem_conf), 0.1)
            self.anchors.append(Anchor(sample.feature.clone(), sem_vec, set(code), conf))
            self._upsert_cluster(sample.feature, sample.prob, sem_vec, code,
                                 sample.reliability * conf)

    def propagate(self, feature: torch.Tensor) -> Tuple[Optional[torch.Tensor], Set[str], float]:
        if len(self.anchors) == 0:
            return None, set(), 0.0
        device = feature.device
        feat = safe_normalize(feature.detach().float(), dim=-1)
        anchor_feats = torch.stack([a.feature for a in self.anchors], dim=0).to(device)
        sims = anchor_feats @ feat
        k = min(self.cfg.anchor_neighbors, sims.numel())
        vals, idx = sims.topk(k=k)
        weights = torch.softmax(vals / max(self.cfg.anchor_temperature, 1e-6), dim=0)
        sems = torch.stack([self.anchors[int(i)].sem for i in idx], dim=0).to(device)
        sem = safe_normalize((sems * weights[:, None]).sum(dim=0), dim=0)
        conf = float(sum(float(weights[j]) * self.anchors[int(idx[j])].confidence for j in range(k)))
        codes: Set[str] = set()
        for j in range(k):
            if float(weights[j]) > 0.15 or j == 0:
                codes.update(self.anchors[int(idx[j])].code)
        return sem, codes, conf

    def retrieve(self, feature: torch.Tensor, prob: torch.Tensor,
                 sem: Optional[torch.Tensor], code: Set[str]) -> Tuple[Optional[int], float]:
        if not self.clusters:
            return None, float("-inf")
        device = feature.device
        feat = safe_normalize(feature.detach().float(), dim=-1)
        prob_detached = normalize_prob(prob.detach().float())
        best_idx: Optional[int] = None
        best_score = float("-inf")
        for idx, cluster in enumerate(self.clusters):
            if cluster.q_rel < self.cfg.tau_q:
                continue
            score = self.cfg.alpha_v * float(feat @ cluster.mu.to(device))
            score += self.cfg.alpha_p * float(prob_detached @ cluster.p_bar.to(device))
            if sem is not None and cluster.s_sem is not None:
                score += self.cfg.alpha_s * float(
                    safe_normalize(sem.detach().float(), dim=0) @ cluster.s_sem.to(device)
                )
            score += self.cfg.alpha_b * jaccard(code, cluster.b_code)
            score -= self.cfg.alpha_age * min(cluster.age / max(self.cfg.window_size, 1), 1.0)
            if cluster.state != "committed":
                score -= 0.05
            if score > best_score:
                best_score = score
                best_idx = idx
        return best_idx, best_score

    def proto_loss(self, features: torch.Tensor, logits: torch.Tensor,
                   reliabilities: Sequence[float]) -> Tuple[torch.Tensor, List[Tuple[int, float, Optional[torch.Tensor], Set[str], float]]]:
        if not self.cfg.enabled or not self.clusters:
            return logits.sum() * 0.0, []
        probs = logits.softmax(dim=1)
        losses = []
        assignments = []
        for i in range(features.shape[0]):
            sem, code, sem_conf = self.propagate(features[i])
            k_star, score = self.retrieve(features[i], probs[i], sem, code)
            assignments.append((k_star if k_star is not None else -1, score, sem, code, sem_conf))
            if k_star is None or score <= self.cfg.tau_assign:
                continue
            cluster = self.clusters[k_star]
            beta = float(reliabilities[i]) * sigmoid((score - self.cfg.tau_assign) / max(self.cfg.tau_beta, 1e-6))
            if beta <= 0:
                continue
            mu = cluster.mu.to(features.device)
            p_bar = normalize_prob(cluster.p_bar.to(features.device))
            feature_loss = 1.0 - F.cosine_similarity(features[i].float(), mu.float(), dim=0)
            pred_loss = F.kl_div(
                torch.log(probs[i].clamp_min(1e-8)),
                p_bar,
                reduction="sum",
            )
            sem_loss = features[i].sum() * 0.0
            if sem is not None and cluster.s_sem is not None:
                sem_loss = 1.0 - F.cosine_similarity(
                    sem.detach().to(features.device).float(),
                    cluster.s_sem.to(features.device).float(),
                    dim=0,
                )
            losses.append(beta * (
                self.cfg.gamma_f * feature_loss
                + self.cfg.gamma_p * pred_loss
                + self.cfg.gamma_s * sem_loss
            ))
        if not losses:
            return logits.sum() * 0.0, assignments
        return torch.stack(losses).mean(), assignments

    def update_from_batch(self, features: torch.Tensor, logits: torch.Tensor,
                          reliabilities: Sequence[float],
                          assignments: Sequence[Tuple[int, float, Optional[torch.Tensor], Set[str], float]]):
        if not self.cfg.enabled:
            return
        probs = logits.softmax(dim=1).detach().float().cpu()
        features_cpu = safe_normalize(features.detach().float(), dim=-1).cpu()
        covered = 0
        for i, assignment in enumerate(assignments):
            k_star, score, sem, code, sem_conf = assignment
            rel = float(reliabilities[i]) * max(float(sem_conf), 0.1)
            if k_star >= 0 and score > self.cfg.tau_assign:
                covered += 1
            if rel <= self.cfg.tau_rel_store:
                continue
            sem_cpu = sem.detach().cpu() if sem is not None else None
            if k_star >= 0 and score > self.cfg.tau_assign:
                self._update_cluster(self.clusters[k_star], features_cpu[i], probs[i],
                                     sem_cpu, code, rel, score)
            else:
                self._upsert_cluster(features_cpu[i], probs[i], sem_cpu, code, rel)
        self.coverage.append(covered / max(len(assignments), 1))
        self._promote_and_evict()
        self.step += 1
        for cluster in self.clusters:
            cluster.age += 1
        for anchor in self.anchors:
            anchor.age += 1

    def fuse_logits(self, logits: torch.Tensor,
                    assignments: Sequence[Tuple[int, float, Optional[torch.Tensor], Set[str], float]]) -> torch.Tensor:
        if self.cfg.update_mode != "memory_only" or not self.clusters:
            return logits
        probs = logits.softmax(dim=1)
        fused = probs.clone()
        for i, assignment in enumerate(assignments):
            k_star, score, _, _, _ = assignment
            if k_star < 0 or score <= self.cfg.tau_assign:
                continue
            p_bar = normalize_prob(self.clusters[k_star].p_bar.to(logits.device))
            omega = self.cfg.beta_proto * sigmoid((score - self.cfg.tau_assign) / max(self.cfg.tau_beta, 1e-6))
            omega = max(0.0, min(0.9, omega))
            fused[i] = normalize_prob((probs[i] ** (1.0 - omega)) * (p_bar ** omega))
        return torch.log(fused.clamp_min(1e-8))

    def _window_drift(self) -> float:
        if not self.window or not self.clusters:
            return 1.0
        recent = list(self.window)[-min(len(self.window), 8):]
        mus = torch.stack([c.mu for c in self.clusters], dim=0)
        drifts = []
        for sample in recent:
            sim = float((mus @ sample.feature).max())
            drifts.append(1.0 - sim)
        return sum(drifts) / len(drifts)

    def _select_diverse_anchors(self) -> List[BufferedSample]:
        if not self.window:
            return []
        logc = math.log(max(self.window[-1].prob.numel(), 2))
        scored = []
        for sample in self.window:
            score = (
                0.35 * (1.0 - min(sample.l_re / max(logc, 1e-6), 2.0))
                + 0.25 * (1.0 - min(sample.l_ri / max(self.cfg.tau_ri_anchor, 1e-6), 2.0))
                + 0.30 * sample.margin
                - 0.10 * sample.flip
                + 0.20 * sample.reliability
            )
            scored.append((score, sample))
        scored.sort(key=lambda x: x[0], reverse=True)
        pool = [sample for _, sample in scored[:max(self.cfg.anchor_batch * 8, self.cfg.anchor_batch)]]
        selected: List[BufferedSample] = []
        while pool and len(selected) < self.cfg.anchor_batch:
            if not selected:
                selected.append(pool.pop(0))
                continue
            selected_feats = torch.stack([s.feature for s in selected], dim=0)
            distances = [1.0 - float((selected_feats @ sample.feature).max()) for sample in pool]
            best = max(range(len(pool)), key=lambda i: distances[i])
            selected.append(pool.pop(best))
        return selected

    def _upsert_cluster(self, feature: torch.Tensor, prob: torch.Tensor,
                        sem: Optional[torch.Tensor], code: Iterable[str], rel: float):
        feature = safe_normalize(feature.detach().float(), dim=0).cpu()
        prob = normalize_prob(prob.detach().float()).cpu()
        sem_cpu = safe_normalize(sem.detach().float(), dim=0).cpu() if sem is not None else None
        k_star, score = self.retrieve(feature.to(self.device), prob.to(self.device),
                                      sem_cpu.to(self.device) if sem_cpu is not None else None,
                                      set(code))
        if k_star is not None and score > self.cfg.tau_assign:
            self._update_cluster(self.clusters[k_star], feature, prob, sem_cpu, set(code), rel, score)
            return
        self.clusters.append(Cluster(feature, prob, sem_cpu, set(code), float(rel), score_sum=0.0))
        self._promote_and_evict()

    def _update_cluster(self, cluster: Cluster, feature: torch.Tensor, prob: torch.Tensor,
                        sem: Optional[torch.Tensor], code: Set[str], rel: float, score: float):
        eta = self.cfg.eta0 * float(rel) / math.sqrt(cluster.n + 1.0)
        eta = max(0.0, min(1.0, eta))
        old_mu = cluster.mu.clone()
        cluster.mu = safe_normalize((1.0 - eta) * cluster.mu + eta * feature, dim=0)
        cluster.p_bar = normalize_prob((1.0 - eta) * cluster.p_bar + eta * prob)
        if sem is not None:
            if cluster.s_sem is None:
                cluster.s_sem = sem
            else:
                cluster.s_sem = safe_normalize((1.0 - eta) * cluster.s_sem + eta * sem, dim=0)
        cluster.b_code.update(code)
        cluster.q_rel = (1.0 - self.cfg.rho_q) * cluster.q_rel + self.cfg.rho_q * float(rel)
        cluster.var = 0.95 * cluster.var + 0.05 * float(1.0 - (old_mu @ feature))
        cluster.n += 1.0
        cluster.use += 1
        cluster.age = 0
        cluster.score_sum += float(score)

    def _promote_and_evict(self):
        for cluster in self.clusters:
            if cluster.state == "candidate" and cluster.n >= self.cfg.confirm_min_count and cluster.q_rel >= self.cfg.tau_q:
                cluster.state = "committed"
        if len(self.clusters) <= self.cfg.max_clusters:
            return
        self.clusters.sort(key=self._cluster_utility, reverse=True)
        del self.clusters[self.cfg.max_clusters:]

    def _cluster_utility(self, cluster: Cluster) -> float:
        return (
            cluster.q_rel
            + 0.10 * math.log1p(cluster.n)
            - 0.05 * min(cluster.age / max(self.cfg.window_size, 1), 1.0)
            - 0.10 * cluster.var
            + (0.05 if cluster.state == "committed" else 0.0)
        )


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 0.0
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union
