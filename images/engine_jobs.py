"""engine_jobs.py — the job entrypoints behind robovla_job (design §6).

The split from engine.py is the ≤500-line house cap: engine.py carries the
recipe constants, the pure (CPU-testable) helpers, the lazy stack boundary,
dataset loading, model plumbing and the official Phase 1 touchpoints; this
module carries the job dispatch — smoke / train / validate / merge /
gpu_selftest / home_selftest — the file-contract writers per job (contract),
and the checkpoint bookkeeping. The GPU stack is imported only through
engine._require_stack (one boundary, §11).
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from robovla.core.errors import RobovlaError

from . import contract, engine


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
    engine._log(f"[{spec['job_id']}] finished ({t})")
    return 0


def _trainable(model):
    return [p for p in model.parameters() if p.requires_grad]


def _setup(spec, torch, transformers, peft, device):
    """Base model + LoRA wrapper. Returns (base, model, processor)."""
    proc = engine._processor(spec["model"], transformers)
    base = engine._load_base(spec["model"], spec["quant"], torch, transformers,
                             peft, device)
    model = peft.get_peft_model(base, engine._lora_cfg(spec, peft))
    return base, model, proc


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


def _dataset_statistics(out_dir: Path, spec: dict, ds_root: Path) -> None:
    """The official dataset_statistics.json (what the official deploy.py /
    predict_action expect next to a fine-tuned model): q01/q99 bounds. Our
    min/max norm stats ARE those bounds — the [0, 1] contract is the same
    affine map as the official BOUNDS normalization, so official tooling
    denormalizes to the same robot units."""
    p = ds_root / "norm_stats.json"
    if not p.is_file():
        return
    d = json.loads(p.read_text())
    stats = {spec["dataset"]: {"action": {"q01": d["min"], "q99": d["max"],
                                          "mask": [True] * contract.ACTION_DIMS}}}
    (out_dir / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))


# -- the jobs -----------------------------------------------------------------------


def _smoke(spec, job_dir, workdir, ds_root):
    """One optimizer step + the predict-action probe (§6, gate M5)."""
    torch, transformers, peft, _bnb = engine._require_stack(
        "torch", "transformers", "peft", "bitsandbytes")
    device = engine._device(torch)
    engine._log(f"[{spec['job_id']}] smoke: loading {spec['model']} ({spec['quant']})")
    _base, model, proc = _setup(spec, torch, transformers, peft, device)
    opt = torch.optim.AdamW(_trainable(model), lr=spec["optim"]["lr"])
    train, held, _stats = engine._samples(ds_root, spec)
    if not train or not held:
        raise RobovlaError("dataset_not_ready", "need train + held-out samples")
    frame, prompt, _n, target = train[0]
    loss = engine._forward_loss(model, torch, device, engine._load_image(frame),
                                prompt, target, proc)
    opt.zero_grad()
    loss.backward()
    opt.step()
    vram_mb, util = engine._vram(torch, device)
    contract.write_metrics(job_dir, contract.metrics_row(
        1, 1, round(float(loss), 4), spec["optim"]["lr"],
        round(_grad_norm(model), 4), vram_mb, util, 0))
    hf, hp, _hn, _ht = held[0]
    contract.write_probe(job_dir, list(engine.probe_from_bins(
        engine._predict_bins(model, torch, device, engine._load_image(hf), hp, proc))))
    engine._log(f"[{spec['job_id']}] smoke: 1 step, loss={float(loss):.4f}, probe published")


def _train(spec, job_dir, workdir, ds_root):
    """The QLoRA training loop: per-step metrics, per-epoch ckpt + val, the
    final adapter artifact, crash resume (spec.resume_from)."""
    torch, transformers, peft, _bnb = engine._require_stack(
        "torch", "transformers", "peft", "bitsandbytes")
    device = engine._device(torch)
    engine._log(f"[{spec['job_id']}] train: {spec['model']} {spec['quant']} "
                f"r={spec['lora']['rank']} a={spec['lora']['alpha']} "
                f"lr={spec['optim']['lr']} epochs={spec['optim']['epochs']} "
                f"batch={spec['optim']['batch']} ga={spec['optim']['grad_accum']}")
    _base, model, proc = _setup(spec, torch, transformers, peft, device)
    if spec["resume_from"]:
        adapter = _adapter_dir(workdir, spec)
        model.load_adapter(adapter, "default")
        engine._log(f"[{spec['job_id']}] resumed from {adapter}")
    train, held, _stats = engine._samples(ds_root, spec)
    if not train:
        raise RobovlaError("dataset_not_ready", "train split is empty")
    batch_n, gacc = spec["optim"]["batch"], spec["optim"]["grad_accum"]
    per = batch_n * gacc
    total_steps = max(1, math.ceil(len(train) / per) * spec["optim"]["epochs"])
    opt = torch.optim.AdamW(_trainable(model), lr=spec["optim"]["lr"])
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
                loss = engine._forward_loss(model, torch, device,
                                            engine._load_image(frame), prompt, target, proc)
                (loss / per).backward()
                loss_sum += float(loss)
                gnorm = _grad_norm(model)
            opt.step()
            n = max(1, min(per, len(train[i:])))
            elapsed = time.monotonic() - t0
            row = contract.metrics_row(
                step, total_steps, round(loss_sum / n, 4), spec["optim"]["lr"],
                round(gnorm, 4), *engine._vram(torch, device),
                int(elapsed / step * (total_steps - step)))
            contract.write_metrics(job_dir, row)
            engine._log(f"[{spec['job_id']}] step {step}/{total_steps} "
                        f"loss={row['loss']:.4f}")
        _save_ckpt(model, job_dir, f"epoch-{epoch + 1}")
        acc = _eval(model, proc, torch, device, held, 200)
        contract.write_val(job_dir, acc)
        engine._log(f"[{spec['job_id']}] epoch {epoch + 1}: val_accuracy={acc:.4f}")
    final = _save_ckpt(model, job_dir, "final")
    _dataset_statistics(final, spec, ds_root)
    contract.write_artifacts(
        job_dir, workdir,
        [("adapter", final / "adapter_model.safetensors")])


def _eval(model, proc, torch, device, held, cap: int) -> float:
    """Held-out accuracy: fraction of samples within EVAL_TOLERANCE
    (per-dim MAE, normalized space — bins/255 units)."""
    if not held:
        return 0.0
    scale = float(contract.ACTION_DIMS) * (contract.ACTION_BINS - 1)
    correct = 0
    for frame, prompt, _n, target in held[:cap]:
        pred = engine._predict_bins(model, torch, device,
                                    engine._load_image(frame), prompt, proc)
        mae = sum(abs(p - t) for p, t in
                  zip(pred, [engine.norm_to_bin(x) for x in target])) / scale
        correct += mae <= engine.EVAL_TOLERANCE
    return correct / min(cap, len(held))


def _validate(spec, job_dir, workdir, ds_root):
    """Held-out evaluation of a checkpoint (auto after train, §6)."""
    torch, transformers, peft, _bnb = engine._require_stack(
        "torch", "transformers", "peft", "bitsandbytes")
    device = engine._device(torch)
    _base, model, proc = _setup(spec, torch, transformers, peft, device)
    adapter = _adapter_dir(workdir, spec)
    if not adapter.is_dir():
        raise RobovlaError("not_found", f"no checkpoint at {adapter}")
    model.load_adapter(adapter, "default")
    _train, held, _stats = engine._samples(ds_root, spec)
    acc = _eval(model, proc, torch, device, held, 10 ** 9)
    contract.write_val(job_dir, acc)
    contract.write_artifacts(job_dir, workdir,
                             [("adapter", adapter / "adapter_model.safetensors")])
    engine._log(f"[{spec['job_id']}] validate: accuracy={acc:.4f} on {adapter.name}")


def _merge(spec, job_dir, workdir, ds_root):
    """merge_and_unload() + the 10-image identity check (§6): the merged
    model must match the adapter model on held-out images."""
    torch, transformers, peft, _bnb = engine._require_stack(
        "torch", "transformers", "peft", "bitsandbytes")
    device = engine._device(torch)
    _base, model, proc = _setup(spec, torch, transformers, peft, device)
    adapter = _adapter_dir(workdir, spec)
    model.load_adapter(adapter, "default")
    _train, held, _stats = engine._samples(ds_root, spec)
    check = held[:engine.MERGE_CHECK_IMAGES]
    if len(check) < engine.MERGE_CHECK_IMAGES:
        raise RobovlaError(
            "dataset_not_ready", f"need {engine.MERGE_CHECK_IMAGES} held-out images")
    before = [engine._predict_bins(model, torch, device, engine._load_image(f), p, proc)
              for f, p, _n, _t in check]
    merged = model.merge_and_unload()
    after = [engine._predict_bins(merged, torch, device, engine._load_image(f), p, proc)
             for f, p, _n, _t in check]
    for b, a in zip(before, after):
        if any(abs(x - y) > engine.MERGE_TOLERANCE_BINS for x, y in zip(b, a)):
            raise RobovlaError("internal", f"merge identity check failed: {b} vs {a}")
    out = job_dir / "ckpt" / "merged"
    merged.save_pretrained(out)  # ~15 GB export (§6; 14-day purge, §8.5)
    _dataset_statistics(out, spec, ds_root)
    contract.write_artifacts(job_dir, workdir, [("merged", out)])
    engine._log(f"[{spec['job_id']}] merge: identity ok ({len(check)} images)")


# -- the selftests (the stack gates, §11) -------------------------------------------


def _selftest(spec, job_dir, workdir, ds_root, home: bool) -> None:
    """``home`` = ROCm dev lab (bf16, no bnb): GEMM benchmark + bf16 load +
    10-step bf16 fwd/bwd. Campus = 8-bit load, 8-bit fwd/bwd, bf16 fwd/bwd
    where it fits, then predict_action probe + 10-step QLoRA when a dataset
    is present."""
    quant = "bf16" if home else "bnb8"
    stack = ("torch", "transformers", "peft",
             "openvla" if home else "bitsandbytes")
    torch, transformers, peft, _x = engine._require_stack(*stack)
    device = engine._device(torch)
    tag = "home_selftest" if home else "gpu_selftest"
    engine._log(f"[{spec['job_id']}] {tag}: {torch.cuda.get_device_name(0)}")
    if home:
        x = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
        t0 = time.monotonic()
        for _ in range(100):
            x @ x
        torch.cuda.synchronize()
        engine._log(f"[{spec['job_id']}] bf16 GEMM 4096^3: "
                    f"{2 * 4096 ** 3 * 100 / (time.monotonic() - t0) / 1e12:.1f} TFLOPS")
    proc = engine._processor(spec["model"], transformers)
    t0 = time.monotonic()
    base = engine._load_base(spec["model"], quant, torch, transformers, peft, device)
    engine._log(f"[{spec['job_id']}] loaded {spec['model']} {quant} in "
                f"{time.monotonic() - t0:.1f} s")
    _fwd_bwd(spec["job_id"], quant, base, torch, device, proc)
    if not home:
        free, _total = torch.cuda.mem_get_info(device)
        if free >= 20 * 2 ** 30:  # bf16 7.5 B fwd+bwd needs headroom
            _fwd_bwd(spec["job_id"], "bf16", base, torch, device, proc)
        else:
            engine._log(f"[{spec['job_id']}] bf16 fwd/bwd skipped "
                        f"(free {free / 2 ** 30:.1f} GB)")
    _selftest_dataset(spec, job_dir, torch, transformers, peft, base, proc,
                      device, ds_root)
    engine._log(f"[{spec['job_id']}] {tag}: OK")


def _selftest_dataset(spec, job_dir, torch, transformers, peft, base, proc,
                      device, ds_root) -> None:
    """predict-action probe + 10-step QLoRA when a dataset is present."""
    if not (ds_root.is_dir() and (ds_root / "episodes").is_dir()):
        engine._log(f"[{spec['job_id']}] no dataset at {ds_root} — predict/QLoRA skipped")
        return
    train, held, stats = engine._samples(ds_root, spec)
    if held:
        f, p, _n, _t = held[0]
        contract.write_probe(job_dir, list(engine.probe_from_bins(
            engine._predict_bins(base, torch, device, engine._load_image(f), p, proc))))
        engine._log(f"[{spec['job_id']}] predict_action probe published")
    if not train:
        return
    model = peft.get_peft_model(base, engine._lora_cfg(spec, peft))
    opt = torch.optim.AdamW(_trainable(model), lr=spec["optim"]["lr"])
    for step in range(1, 11):
        frame, prompt, _n, target = train[(step - 1) % len(train)]
        loss = engine._forward_loss(model, torch, device, engine._load_image(frame),
                                    prompt, target, proc)
        opt.zero_grad()
        loss.backward()
        opt.step()
        row = contract.metrics_row(step, 10, round(float(loss), 4),
                                   spec["optim"]["lr"],
                                   round(_grad_norm(model), 4),
                                   *engine._vram(torch, device), 0)
        contract.write_metrics(job_dir, row)
        engine._log(f"[{spec['job_id']}] selftest step {step}/10 "
                    f"loss={row['loss']:.4f}")


def _fwd_bwd(job_id: str, tag: str, base, torch, device, proc) -> None:
    """One fwd/bwd pass on a dummy sample (keeps the selftest dataset-free)."""
    from PIL import Image

    dummy = Image.new("RGB", (224, 224), (128, 128, 128))
    prompt = "In: What action should the robot take to selftest?\nOut:"
    loss = engine._forward_loss(base, torch, device, dummy, prompt,
                                engine.probe_from_bins((127,) * 7), proc)
    loss.backward()
    engine._log(f"[{job_id}] fwd/bwd {tag}: loss={float(loss):.4f}")
