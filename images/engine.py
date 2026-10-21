"""engine.py — the training / selftest engine behind robovla_job (design §6).

Recipe (all train-class jobs): 8-bit base (bnb8) or bf16 no-quant (home),
PEFT LoRA + lr/epochs/batch/grad-accum from the spec, the official OpenVLA
prompt + action-token construction (the job entrypoints live in
engine_jobs.py — the ≤500-line house cap). 7-dim action space (256 bins,
[0, 1] contract convention, dataset norm stats).

Phase 1 (verified on silicon, host65/RTX 4070, 2026-10-21): the official
model (``openvla/openvla-7b``) is a HF CUSTOM-CODE model —
AutoModelForVision2Seq/AutoProcessor with trust_remote_code; the prismatic
repo (PINNED commit — vla_compat.py) supplies the prompt builder + the
ActionTokenizer math. The GPU stack (torch/transformers/peft/bitsandbytes)
is imported LAZILY in :func:`_require_stack` — the single GPU-pin site; a
broken stack fails with a clean §9.6 ``internal`` error, never a traceback.

The prompt builder is VENDORed (images/vla_compat.py): ``import prismatic``
drags draccus + tensorflow. The action-token math below is pure (no numpy)
and mirrors the official ActionTokenizer EXACTLY (probe cross-check, 200
random values): v in [0,1] -> bin floor(255 v) -> token id vocab-1-bin (the
last 256 vocab tokens); decode is the inverse, edge-clamped. The action
token string re-tokenizes with the "" token (EMPTY_TOKEN_ID) prefixed — the
predict path appends it before generate, as the official predict_action does.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from robovla.core.errors import RobovlaError
from robovla.core.norm_stats import NormStats, normalize
from robovla.core.split import split
from robovla.episode import to_action_7d

from . import contract  # noqa: F401  (re-exported for the entrypoint's docs)
from .vla_compat import PurePromptBuilder

#: LoRA target modules (the LM linear layers, per the OpenVLA recipe).
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
EVAL_TOLERANCE = 0.10        #: per-dim MAE (normalized) for a held-out sample to be "correct"
MERGE_CHECK_IMAGES = 10      #: held-out images for the merge identity check
MERGE_TOLERANCE_BINS = 1     #: per-dim bin tolerance for that check

#: The "" (empty) token in the Llama-2-7b-hf vocab (id 29871). The official
#: predict_action hardcodes it; it prefixes the re-tokenized action tokens
#: and the predict path appends it before generate. Phase 1 probe:
#: decode([29871]) == "" and tok("") is NOT a derivation (it returns BOS).
EMPTY_TOKEN_ID = 29871

_HUMAN_WRAP = re.compile(r"^In:\s*(?P<msg>.*?)\s*\nOut:\s*$", re.S)

_PROCESSORS: dict[str, Any] = {}


# -- pure helpers (CPU-testable) ---------------------------------------------------


def action_bins(norm7, bins: int = contract.ACTION_BINS) -> tuple[int, ...]:
    """7-dim [0, 1] normalized action -> 256 bins; clamps (0 -> 0, 1 -> bins-1)."""
    vals = contract._finite_floats(norm7, "action")
    return tuple(int(round(max(0.0, min(1.0, v)) * (bins - 1))) for v in vals)


def probe_from_bins(bins, bins_n: int = contract.ACTION_BINS) -> tuple[float, ...]:
    """Inverse of :func:`action_bins` — the [0, 1] values probe.json carries
    (the Pi denormalizes with the dataset stats, §9.5)."""
    return tuple(b / (bins_n - 1) for b in bins)


def norm_to_bin(v: float, bins: int = contract.ACTION_BINS) -> int:
    """A [0, 1] value -> the OFFICIAL action bin: floor((bins-1) v), clamped —
    the official ActionTokenizer's digitize on linspace(-1,1,bins) of u = 2v-1
    (the Phase 1 probe cross-checked the two mappings: 200/200 match)."""
    return max(0, min(bins - 1, math.floor((bins - 1) * max(0.0, min(1.0, v)))))


def norm_to_token_id(v: float, vocab_size: int,
                     bins: int = contract.ACTION_BINS) -> int:
    """A [0, 1] value -> the official action token id: the last ``bins`` vocab
    tokens, id = vocab - 1 - bin (bin from :func:`norm_to_bin`)."""
    return vocab_size - 1 - norm_to_bin(v, bins)


def token_id_to_bin(token_id: int, vocab_size: int,
                    bins: int = contract.ACTION_BINS) -> int:
    """Official action token id -> bin (the inverse of
    :func:`norm_to_token_id`); out-of-range ids clamp to the edge bin — the
    official decode's clipping semantics."""
    return max(0, min(bins - 1, vocab_size - 1 - token_id))


def _human_msg(template: str) -> str:
    """The bare human message behind the model-card wrap (the spec template
    is the WRAPPED form; the official builder re-wraps, so extraction keeps
    training strings byte-identical). Non-canonical templates pass through."""
    m = _HUMAN_WRAP.match(template)
    return m.group("msg") if m else template


def _log(msg: str) -> None:
    # stdout is block-buffered when the agent redirects it to the job log:
    # flush per line so the supervisor's log tail is ordered.
    print(msg, flush=True)


# -- the lazy stack boundary --------------------------------------------------------


def _require_stack(*names: str):
    """Import the pinned stack (the single GPU-pin site, design §6); a
    missing module = broken image = clean §9.6 ``internal``, never a bare
    ImportError traceback."""
    import importlib

    mods = []
    for name in names:
        try:
            mods.append(importlib.import_module(name))
        except ImportError as e:
            raise RobovlaError(
                "internal",
                f"runner stack unavailable: {e.name or name} — "
                f"is the runner image broken?") from e
    return mods


def _device(torch):
    if not torch.cuda.is_available():
        raise RobovlaError("internal", "no CUDA device visible (GPU passthrough?)")
    return torch.device("cuda")


# -- dataset loading (the agent-pulled bundle under <workdir>/datasets) -------------


def _episodes_raw(ds_root: Path):
    """(episode_id, task, [frame_paths], [7d actions]) from episodes/ep_*/."""
    # load_episode is the FILE SEAM (fs) — it parses episode.json/actions.json
    # and counts frames/; the pure core (robovla.episode) carries no I/O.
    from robovla.fs import load_episode

    root = ds_root / "episodes"
    if not root.is_dir():
        raise RobovlaError("dataset_not_ready", f"no episodes in {ds_root}")
    out = []
    for d in sorted(root.iterdir()):
        if not d.is_dir():
            continue
        ep = load_episode(d)
        frames = [str(f) for f in sorted(d.glob("frame_*.jpg"))]
        if not frames:
            raise RobovlaError("dataset_not_ready", f"episode {ep.id} has no frames")
        out.append((ep.id, ep.task, frames, [to_action_7d(a) for a in ep.actions]))
    return out


def _samples(ds_root: Path, spec: dict):
    """(train, held_out) sample lists + dataset NormStats; one sample per
    (frame, action) pair, normalized with the dataset's stats; the held-out
    partition is the SHARED seeded split (§5.2). Sample = (frame_path,
    prompt_wrapped, normalized_7d, normalized_7d)."""
    p = ds_root / "norm_stats.json"
    if not p.is_file():
        raise RobovlaError("dataset_not_ready", f"no norm stats in {ds_root}")
    stats = NormStats(min=tuple(json.loads(p.read_text())["min"]),
                      max=tuple(json.loads(p.read_text())["max"]))
    eps = _episodes_raw(ds_root)
    s = split([e[0] for e in eps], spec["student"], spec["dataset"].split("/")[1])

    def pick(held: bool):
        out = []
        for eid, task, frames, acts in eps:
            if (eid in s.held_out_set) != held:
                continue
            prompt = spec["prompt_template"].format(task=task)
            for a, frame in zip(acts, frames):
                n = normalize(a, stats)
                out.append((frame, prompt, n, tuple(n)))
        return out

    return pick(False), pick(True), stats


def _load_image(path: str):
    """224^2 RGB (the prep seam's output size) as a PIL image."""
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if img.size != (224, 224):
        img = img.resize((224, 224))
    return img


# -- model plumbing -----------------------------------------------------------------


def _processor(model_id: str, transformers):
    """The HF processor (tokenizer + image processor), custom-code, cached."""
    if model_id not in _PROCESSORS:
        _PROCESSORS[model_id] = transformers.AutoProcessor.from_pretrained(
            model_id, trust_remote_code=True)
    return _PROCESSORS[model_id]


def _load_base(model_id: str, quant: str, torch, transformers, peft, device):
    """The base VLA: 8-bit (bnb8, campus) or bf16 no-quant (home). bnb8
    follows the official k-bit QLoRA path (BitsAndBytesConfig load_in_8bit +
    prepare_model_for_kbit_training); bf16 moves to the device explicitly."""
    if quant not in ("bnb8", "bf16"):
        raise RobovlaError("validation", f"unknown quant profile: {quant!r}")
    kwargs = dict(torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
                  trust_remote_code=True)
    if quant == "bnb8":
        kwargs["quantization_config"] = transformers.BitsAndBytesConfig(
            load_in_8bit=True)
    model = transformers.AutoModelForVision2Seq.from_pretrained(model_id, **kwargs)
    return peft.prepare_model_for_kbit_training(model) if quant == "bnb8" \
        else model.to(device)


def _lora_cfg(spec: dict, peft):
    return peft.LoraConfig(
        r=spec["lora"]["rank"], lora_alpha=spec["lora"]["alpha"],
        lora_dropout=0.0, target_modules=list(LORA_TARGETS),
        init_lora_weights="gaussian", inference_mode=False)


def _vram(torch, device) -> tuple[int, int]:
    """(allocated MB, utilization % of the device's memory)."""
    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    return int(used / 1e6), int(100.0 * used / max(1, total))


# -- prompt + input construction (the official Phase 1 touchpoints) -----------------


def _action_token_str(proc, v7) -> str:
    """The gpt-turn action token string: decode of the official action token
    ids (re-tokenizes with the "" token prefixed — the training-time input)."""
    V = proc.tokenizer.vocab_size
    return proc.tokenizer.decode([norm_to_token_id(v, V) for v in v7])


def _train_prompt(proc, prompt_wrapped: str, v7) -> str:
    """The full training prompt: the official PurePromptBuilder turns
    (human: task message, gpt: action token string + eos)."""
    pb = PurePromptBuilder("openvla")
    pb.add_turn("human", _human_msg(prompt_wrapped))
    pb.add_turn("gpt", _action_token_str(proc, v7))
    return pb.get_prompt()


def _train_inputs(proc, torch, device, image, prompt_wrapped: str, v7) -> dict:
    """One sample's inputs in the official shapes: input_ids / attention_mask
    / pixel_values (bf16, [1,3,224,224]) / labels (-100 everywhere but the 7
    action tokens + the stop token)."""
    ids = proc.tokenizer(_train_prompt(proc, prompt_wrapped, v7),
                         add_special_tokens=True).input_ids
    labels = torch.tensor(ids)  # tensor FIRST: the extended-slice mask needs it
    labels[: -(len(v7) + 1)] = -100
    pv = proc.image_processor.apply_transform(image)
    return dict(
        input_ids=torch.tensor([ids], device=device),
        attention_mask=torch.ones((1, len(ids)), dtype=torch.long, device=device),
        pixel_values=torch.as_tensor(pv, dtype=torch.bfloat16).unsqueeze(0).to(device),
        labels=labels.unsqueeze(0).to(device))


def _forward_loss(model, torch, device, image, prompt_wrapped: str, v7, proc):
    """One sample's loss (the official training forward: bf16 autocast)."""
    inputs = _train_inputs(proc, torch, device, image, prompt_wrapped, v7)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(**inputs).loss


def _predict_bins(model, torch, device, image, prompt_wrapped: str, proc):
    """The 7 predicted action BINS (contract: bin b <-> value b/255). The
    official predict path (Phase 1): the bare wrapped prompt -> append the
    "" token (EMPTY_TOKEN_ID) if the prompt doesn't already end with it (the
    training-time input) -> greedy generate(7) -> token ids -> bins."""
    V = proc.tokenizer.vocab_size
    model.eval()
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            # autocast: the bnb8 model keeps fp32 bias/LN on the (unquantized)
            # vision convs — the training forwards run under bf16 autocast, so
            # generate must too or the conv2d dtype check fires
            pin = proc(prompt_wrapped, image, return_tensors="pt").to(device)
            in_ids, in_mask = pin["input_ids"], pin["attention_mask"]
            in_pv = pin["pixel_values"].to(torch.bfloat16)
            if in_ids[0, -1].item() != EMPTY_TOKEN_ID:
                in_ids = torch.cat((in_ids, in_ids.new_full((1, 1), EMPTY_TOKEN_ID)),
                                   dim=1)
                in_mask = torch.cat(
                    (in_mask, torch.ones((1, 1), dtype=in_mask.dtype,
                                         device=device)), dim=1)
            gen = model.generate(input_ids=in_ids, attention_mask=in_mask,
                                 pixel_values=in_pv, max_new_tokens=7,
                                 do_sample=False)
    finally:
        model.train()
    return [token_id_to_bin(i, V) for i in gen[0, -7:].tolist()]
