"""engine.py — the training / selftest engine behind robovla_job (design §6).

Recipe (all train-class jobs): 8-bit base (bnb8) or bf16 no-quant (home),
PEFT LoRA + lr/epochs/batch/grad-accum from the spec, ``max_seq_len`` 287,
7-dim action space (256 bins, [0, 1] convention, dataset norm stats).

The GPU stack (torch/transformers/peft/bitsandbytes/openvla) is imported
LAZILY in :func:`_require_stack` — the single GPU-pin site (finalized in
Phase 1 at the first gpu_selftest; the official finetune.py crosscheck
guards recipe correctness). A broken stack fails with a clean §9.6
``internal`` error. Pure helpers carry no stack dependency, so the CPU
suite checks the binning and the boundary.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

from robovla.core.errors import RobovlaError
from robovla.core.norm_stats import NormStats, normalize
from robovla.core.split import split
from robovla.episode import to_action_7d

from . import contract  # noqa: F401  (re-exported for the entrypoint's docs)

#: LoRA target modules (the LM linear layers, per the OpenVLA recipe).
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj")
EVAL_TOLERANCE = 0.10        #: per-dim MAE (normalized) for a held-out sample to be "correct"
MERGE_CHECK_IMAGES = 10      #: held-out images for the merge identity check
MERGE_TOLERANCE_BINS = 1     #: per-dim bin tolerance for that check

#: NOTE(Phase 1): the ``openvla.*`` touchpoints follow the PINNED commit —
#: verified at the first gpu_selftest (the stack gate, §11).

_TOKENIZERS: dict[str, Any] = {}


# -- pure helpers (CPU-testable) ---------------------------------------------------


def action_bins(norm7, bins: int = contract.ACTION_BINS) -> tuple[int, ...]:
    """7-dim normalized action (each dim in [0, 1], core.norm_stats) ->
    256 bins. Clamps to range: 0 -> 0, 1 -> bins-1 (the 256-bin x 7-token
    model-card action space)."""
    vals = contract._finite_floats(norm7, "action")
    return tuple(int(round(max(0.0, min(1.0, v)) * (bins - 1))) for v in vals)


def probe_from_bins(bins, bins_n: int = contract.ACTION_BINS) -> tuple[float, ...]:
    """Inverse of :func:`action_bins`: the [0, 1] normalized action a
    prediction corresponds to — the value probe.json carries (the Pi
    denormalizes with the dataset stats, §9.5)."""
    return tuple(b / (bins_n - 1) for b in bins)


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
    from robovla.episode import load_episode

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
    """(train, held_out) sample lists + dataset NormStats. One sample per
    (frame, action) pair, normalized with the dataset's stats; the held-out
    partition is the SHARED seeded split (§5.2)."""
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


def _load_base(model_id: str, quant: str, torch, transformers):
    """The base VLA: 8-bit (bnb8) or bf16 no-quant (home preset)."""
    if quant == "bnb8":
        return transformers.LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.float16, load_in_8bit=True,
            device_map="auto")
    if quant == "bf16":
        return transformers.LlavaForConditionalGeneration.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
    raise RobovlaError("validation", f"unknown quant profile: {quant!r}")


def _tokenizer(model_id: str, transformers):
    if model_id not in _TOKENIZERS:
        _TOKENIZERS[model_id] = transformers.AutoTokenizer.from_pretrained(model_id)
    return _TOKENIZERS[model_id]


def _lora_cfg(spec: dict, peft):
    return peft.LoraConfig(
        r=spec["lora"]["rank"], lora_alpha=spec["lora"]["alpha"],
        target_modules=list(LORA_TARGETS), task_type="CAUSAL_LM",
        inference_mode=False)


def _vram(torch, device) -> tuple[int, int]:
    """(allocated MB, utilization % of the device's memory)."""
    free, total = torch.cuda.mem_get_info(device)
    used = total - free
    return int(used / 1e6), int(100.0 * used / max(1, total))


def _inputs(model, torch, device, image, prompt, bins, tokenizer):
    """Prompt collation + action targets (the pinned openvla touchpoint)."""
    import openvla

    inputs = openvla.tokenize_prompt(tokenizer, prompt, image, max_length=287)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    inputs["action_target"] = torch.tensor([bins], device=device)
    return inputs


def _forward_loss(model, torch, device, image, prompt, bins, tokenizer):
    return model(**_inputs(model, torch, device, image, prompt, bins, tokenizer)).loss


def _predict_bins(model, torch, device, image, prompt, tok,
                  bins: int = contract.ACTION_BINS):
    """The 7 predicted action bins for one image + prompt (generate, no grad)."""
    import openvla

    model.eval()
    try:
        with torch.no_grad():
            out = model.generate(**{k: v for k, v in
                                    _inputs(model, torch, device, image, prompt,
                                            [0] * 7, tok).items()
                                    if k != "action_target"},
                                 max_new_tokens=7)
    finally:
        model.train()
    return [openvla.action_token_to_bin(i, bins) for i in out[0, -7:].tolist()]


def _grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().norm() ** 2)
    return math.sqrt(total)


def _save_ckpt(model, job_dir, name) -> Path:
    d = job_dir / "ckpt" / name
    d.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(d)
    return d


def _adapter_dir(workdir, spec) -> Path:
    """The adapter dir for resume/validate/merge under this run's ckpt/."""
    return workdir / "runs" / spec["job_id"] / "ckpt" / (spec["resume_from"] or "final")


# -- job entrypoints ------------------------------------------------------------------


def run_job(spec: dict, spec_path: str) -> int:
    job_dir, workdir = contract.job_dirs(spec_path)
    ds_root = workdir / "datasets" / spec["dataset"].replace("/", "__")
    t = spec["type"]
    if t == "smoke":
        _smoke(spec, job_dir, workdir, ds_root)
    elif t == "train":
        _train(spec, job_dir, workdir, ds_root)
    elif t == "validate":
        _validate(spec, job_dir, workdir, ds_root)
    elif t == "merge":
        _merge(spec, job_dir, workdir, ds_root)
    elif t == "gpu_selftest":
        _selftest(spec, job_dir, workdir, ds_root, home=False)
    elif t == "home_selftest":
        _selftest(spec, job_dir, workdir, ds_root, home=True)
    else:  # defensive: the entrypoint already rejects everything else
        raise RobovlaError("validation", f"unhandled job type: {t!r}")
    _log(f"[{spec['job_id']}] finished ({t})")
    return 0


def _setup(spec, torch, transformers, peft, device):
    """Base model + (optional) LoRA wrapper. Returns (base, model, tokenizer)."""
    base = _load_base(spec["model"], spec["quant"], torch, transformers)
    tok = _tokenizer(spec["model"], transformers)
    base._tok = tok  # the collator's tokenizer (single site)
    model = peft.get_peft_model(base, _lora_cfg(spec, peft))
    return base, model, tok


def _smoke(spec, job_dir, workdir, ds_root):
    """One optimizer step + the predict-action probe (§6, gate M5)."""
    torch, transformers, peft, _bnb = _require_stack(
        "torch", "transformers", "peft", "bitsandbytes")
    device = _device(torch)
    _log(f"[{spec['job_id']}] smoke: loading {spec['model']} ({spec['quant']})")
    _base, model, tok = _setup(spec, torch, transformers, peft, device)
    opt = torch.optim.AdamW(model.parameters(), lr=spec["optim"]["lr"])
    train, held, stats = _samples(ds_root, spec)
    if not train or not held:
        raise RobovlaError("dataset_not_ready", "need train + held-out samples")
    frame, prompt, _n, target = train[0]
    loss = _forward_loss(model, torch, device, _load_image(frame), prompt,
                         action_bins(target), tok)
    opt.zero_grad()
    loss.backward()
    opt.step()
    vram_mb, util = _vram(torch, device)
    contract.write_metrics(job_dir, contract.metrics_row(
        1, 1, round(float(loss), 4), spec["optim"]["lr"],
        round(_grad_norm(model), 4), vram_mb, util, 0))
    hf, hp, _hn, _ht = held[0]
    contract.write_probe(job_dir, list(probe_from_bins(
        _predict_bins(model, torch, device, _load_image(hf), hp, tok))))
    _log(f"[{spec['job_id']}] smoke: 1 step, loss={float(loss):.4f}, probe published")


def _train(spec, job_dir, workdir, ds_root):
    """The QLoRA training loop: per-step metrics, per-epoch ckpt + val, the
    final adapter artifact, crash resume (spec.resume_from)."""
    torch, transformers, peft, _bnb = _require_stack(
        "torch", "transformers", "peft", "bitsandbytes")
    device = _device(torch)
    _log(f"[{spec['job_id']}] train: {spec['model']} {spec['quant']} "
         f"r={spec['lora']['rank']} a={spec['lora']['alpha']} "
         f"lr={spec['optim']['lr']} epochs={spec['optim']['epochs']} "
         f"batch={spec['optim']['batch']} ga={spec['optim']['grad_accum']}")
    _base, model, tok = _setup(spec, torch, transformers, peft, device)
    if spec["resume_from"]:
        adapter = _adapter_dir(workdir, spec)
        model.load_adapter(adapter, "default")
        _log(f"[{spec['job_id']}] resumed from {adapter}")
    train, held, _stats = _samples(ds_root, spec)
    if not train:
        raise RobovlaError("dataset_not_ready", "train split is empty")
    batch_n, gacc = spec["optim"]["batch"], spec["optim"]["grad_accum"]
    per = batch_n * gacc
    total_steps = max(1, math.ceil(len(train) / per) * spec["optim"]["epochs"])
    opt = torch.optim.AdamW(model.parameters(), lr=spec["optim"]["lr"])
    t0 = time.monotonic()
    step = 0
    for epoch in range(spec["optim"]["epochs"]):
        for i in range(0, len(train), per):
            step += 1
            opt.zero_grad()
            loss_sum, gnorm = 0.0, 0.0
            for j in range(0, per, batch_n):
                chunk = train[i + j: i + j + batch_n]
                if not chunk:
                    break
                frame, prompt, _n, target = chunk[0]  # single-sample micro-batch
                loss = _forward_loss(model, torch, device, _load_image(frame),
                                     prompt, action_bins(target), tok)
                (loss / per).backward()
                loss_sum += float(loss)
                gnorm = _grad_norm(model)
            opt.step()
            n = max(1, min(per, len(train[i:])))
            elapsed = time.monotonic() - t0
            row = contract.metrics_row(
                step, total_steps, round(loss_sum / n, 4), spec["optim"]["lr"],
                round(gnorm, 4), *_vram(torch, device),
                int(elapsed / step * (total_steps - step)))
            contract.write_metrics(job_dir, row)
            _log(f"[{spec['job_id']}] step {step}/{total_steps} loss={row['loss']:.4f}")
        _save_ckpt(model, job_dir, f"epoch-{epoch + 1}")
        acc = _eval(model, tok, torch, device, held, 200)
        contract.write_val(job_dir, acc)
        _log(f"[{spec['job_id']}] epoch {epoch + 1}: val_accuracy={acc:.4f}")
    _save_ckpt(model, job_dir, "final")
    contract.write_artifacts(
        job_dir, workdir,
        [("adapter", job_dir / "ckpt" / "final" / "adapter_model.safetensors")])


def _eval(model, tok, torch, device, held, cap: int) -> float:
    """Held-out accuracy: fraction of samples within EVAL_TOLERANCE
    (per-dim MAE, normalized space)."""
    if not held:
        return 0.0
    correct = 0
    for frame, prompt, _n, target in held[:cap]:
        pred = [b / (contract.ACTION_BINS - 1) for b in
                _predict_bins(model, torch, device, _load_image(frame), prompt, tok)]
        mae = sum(abs(p - t) for p, t in zip(pred, target)) / 7.0
        correct += mae <= EVAL_TOLERANCE
    return correct / min(cap, len(held))


def _validate(spec, job_dir, workdir, ds_root):
    """Held-out evaluation of a checkpoint (auto after train, §6)."""
    torch, transformers, peft, _bnb = _require_stack(
        "torch", "transformers", "peft", "bitsandbytes")
    device = _device(torch)
    _base, model, tok = _setup(spec, torch, transformers, peft, device)
    adapter = _adapter_dir(workdir, spec)
    if not adapter.is_dir():
        raise RobovlaError("not_found", f"no checkpoint at {adapter}")
    model.load_adapter(adapter, "default")
    _train, held, _stats = _samples(ds_root, spec)
    acc = _eval(model, tok, torch, device, held, 10 ** 9)
    contract.write_val(job_dir, acc)
    contract.write_artifacts(job_dir, workdir,
                             [("adapter", adapter / "adapter_model.safetensors")])
    _log(f"[{spec['job_id']}] validate: accuracy={acc:.4f} on {adapter.name}")


def _merge(spec, job_dir, workdir, ds_root):
    """merge_and_unload() + the 10-image identity check (§6): the merged
    model must match the adapter model on held-out images."""
    torch, transformers, peft, _bnb = _require_stack(
        "torch", "transformers", "peft", "bitsandbytes")
    device = _device(torch)
    _base, model, tok = _setup(spec, torch, transformers, peft, device)
    adapter = _adapter_dir(workdir, spec)
    model.load_adapter(adapter, "default")
    _train, held, _stats = _samples(ds_root, spec)
    check = held[:MERGE_CHECK_IMAGES]
    if len(check) < MERGE_CHECK_IMAGES:
        raise RobovlaError(
            "dataset_not_ready", f"need {MERGE_CHECK_IMAGES} held-out images")
    before = [_predict_bins(model, torch, device, _load_image(f), p, tok)
              for f, p, _n, _t in check]
    merged = model.merge_and_unload()
    after = [_predict_bins(merged, torch, device, _load_image(f), p, tok)
             for f, p, _n, _t in check]
    for b, a in zip(before, after):
        if any(abs(x - y) > MERGE_TOLERANCE_BINS for x, y in zip(b, a)):
            raise RobovlaError(
                "internal", f"merge identity check failed: {b} vs {a}")
    out = job_dir / "ckpt" / "merged"
    merged.save_pretrained(out)  # ~15 GB export (§6; 14-day purge, §8.5)
    contract.write_artifacts(job_dir, workdir, [("merged", out)])
    _log(f"[{spec['job_id']}] merge: identity ok ({len(check)} images)")


def _selftest(spec, job_dir, workdir, ds_root, home: bool) -> None:
    """The stack gates (§11). ``home`` = ROCm dev lab (bf16, no bnb):
    GEMM benchmark + bf16 load + 10-step bf16 fwd/bwd. Campus = 8-bit load,
    8-bit fwd/bwd, bf16 fwd/bwd where it fits, then predict_action probe +
    10-step QLoRA when a dataset is present."""
    quant = "bf16" if home else "bnb8"
    stack = ("torch", "transformers", "peft",
             "openvla" if home else "bitsandbytes")
    torch, transformers, peft, _x = _require_stack(*stack)
    device = _device(torch)
    tag = "home_selftest" if home else "gpu_selftest"
    _log(f"[{spec['job_id']}] {tag}: {torch.cuda.get_device_name(0)}")
    if home:
        x = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        t0 = time.monotonic()
        for _ in range(100):
            x @ x
        torch.cuda.synchronize()
        _log(f"[{spec['job_id']}] bf16 GEMM 4096^3: "
             f"{2 * 4096 ** 3 * 100 / (time.monotonic() - t0) / 1e12:.1f} TFLOPS")
    t0 = time.monotonic()
    base = _load_base(spec["model"], quant, torch, transformers)
    tok = _tokenizer(spec["model"], transformers)
    base._tok = tok
    _log(f"[{spec['job_id']}] loaded {spec['model']} {quant} in "
         f"{time.monotonic() - t0:.1f} s")
    _fwd_bwd(spec["job_id"], quant, base, torch, device, tok)
    if not home:
        free, _total = torch.cuda.mem_get_info(device)
        if free >= 20 * 2 ** 30:  # bf16 7.5 B fwd+bwd needs headroom
            _fwd_bwd(spec["job_id"], "bf16", base, torch, device, tok)
        else:
            _log(f"[{spec['job_id']}] bf16 fwd/bwd skipped "
                 f"(free {free / 2 ** 30:.1f} GB)")
    _selftest_dataset(spec, job_dir, torch, transformers, peft, base, tok,
                      device, ds_root)
    _log(f"[{spec['job_id']}] {tag}: OK")


def _selftest_dataset(spec, job_dir, torch, transformers, peft, base, tok,
                      device, ds_root) -> None:
    """predict-action probe + 10-step QLoRA when a dataset is present."""
    if not (ds_root.is_dir() and (ds_root / "episodes").is_dir()):
        _log(f"[{spec['job_id']}] no dataset at {ds_root} — predict/QLoRA skipped")
        return
    train, held, stats = _samples(ds_root, spec)
    if held:
        f, p, _n, _t = held[0]
        contract.write_probe(job_dir, list(probe_from_bins(
            _predict_bins(base, torch, device, _load_image(f), p, tok))))
        _log(f"[{spec['job_id']}] predict_action probe published")
    if not train:
        return
    model = peft.get_peft_model(base, _lora_cfg(spec, peft))
    opt = torch.optim.AdamW(model.parameters(), lr=spec["optim"]["lr"])
    for step in range(1, 11):
        frame, prompt, _n, target = train[(step - 1) % len(train)]
        loss = _forward_loss(model, torch, device, _load_image(frame), prompt,
                             action_bins(target), tok)
        opt.zero_grad()
        loss.backward()
        opt.step()
        row = contract.metrics_row(step, 10, round(float(loss), 4),
                                   spec["optim"]["lr"],
                                   round(_grad_norm(model), 4),
                                   *_vram(torch, device), 0)
        contract.write_metrics(job_dir, row)
        _log(f"[{spec['job_id']}] selftest step {step}/10 loss={row['loss']:.4f}")


def _fwd_bwd(job_id: str, tag: str, base, torch, device, tok) -> None:
    """One fwd/bwd pass on a dummy sample (keeps the selftest dataset-free)."""
    from PIL import Image

    dummy = Image.new("RGB", (224, 224), (128, 128, 128))
    prompt = "In: What action should the robot take to selftest?\nOut:"
    loss = _forward_loss(base, torch, device, dummy, prompt, [127] * 7, tok)
    loss.backward()
    _log(f"[{job_id}] fwd/bwd {tag}: loss={float(loss):.4f}")
