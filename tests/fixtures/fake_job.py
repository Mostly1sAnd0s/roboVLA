"""A REAL job process for agent integration tests (CPU, no GPU).

The same contract as the image's ``robovla_job.py`` (Phase 1): the only input
is the validated ``--spec <spec.json>`` (fixed argv, no shell); the job
writes the full observability contract under its job dir:

    runs/<job_id>/
      metrics.jsonl    one JSON row per step (the SSE metrics event source)
      (stdout/stderr)  the agent redirects to runs/<job_id>/log
      val.json         {"accuracy": x} per checkpoint (the SSE val event)
      probe.json       [7 floats] — the smoke predict-action probe (§6)
      artifacts.json   [{"kind", "path" workdir-relative}] — the manifest

Behavior is driven by spec fields only (deterministic):
  steps  = 2 * epochs
  seed 999 -> fail after step 1 (exit 1, error line on stderr)
  seed 888 -> ignore SIGTERM, keep sleeping (forces the agent's 10 s grace
              -> SIGKILL path, §8.3)

Step pacing is the ROBOVLA_FAKE_JOB_STEP_S env (default 0.01 s) so the whole
suite stays fast while polls observe real transitions.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import time

ARTIFACT_KINDS = {
    "train": "adapter",
    "validate": "adapter",
    "crosscheck": "adapter",
    "merge": "merged",
}


def main() -> int:
    args = sys.argv[1:]
    if "--spec" not in args:
        print("fake_job: --spec required", file=sys.stderr)
        return 2
    spec_path = args[args.index("--spec") + 1]
    with open(spec_path) as f:
        spec = json.load(f)

    job_type = spec["type"]
    seed = spec["seed"]
    epochs = spec["optim"]["epochs"]
    job_id = spec["job_id"]
    job_dir = os.path.dirname(spec_path)
    workdir = os.path.dirname(os.path.dirname(job_dir))  # workdir/runs/<job_id>
    step_s = float(os.environ.get("ROBOVLA_FAKE_JOB_STEP_S", "0.01"))

    if seed == 888:  # kill-mode: ignore SIGTERM (grace expires -> SIGKILL)
        signal.signal(signal.SIGTERM, lambda *a: None)

    total = max(2, 2 * epochs)
    with open(os.path.join(job_dir, "metrics.jsonl"), "a") as m:
        for step in range(1, total + 1):
            if seed == 999 and step > 1:
                print(f"[{job_id}] OOM: scripted failure (seed 999)", file=sys.stderr)
                return 1
            frac = step / total
            row = {
                "step": step, "total": total,
                "loss": round(2.0 * (1.0 - frac) + 0.05, 4),
                "lr": spec["optim"]["lr"],
                "grad_norm": round(1.5 - frac, 4),
                "vram_mb": 15000, "vram_util_pct": 78,
                "eta_s": int(300 * (1.0 - frac)),
            }
            m.write(json.dumps(row) + "\n")
            m.flush()
            # stdout is block-buffered when redirected to the log file: flush
            # per step so the agent's log tail (and failure reasons) see lines
            # in order — stderr (unbuffered) would otherwise interleave first.
            print(f"[step {step}/{total}] loss={row['loss']} job={job_id}",
                  flush=True)
            if step == total // 2 or step == total:  # checkpoints
                with open(os.path.join(job_dir, "val.json"), "w") as vf:
                    json.dump({"accuracy": round(0.5 + 0.05 * step, 4)}, vf)
            time.sleep(step_s)

    if job_type == "smoke":  # the predict-action probe (§6, gate M5)
        with open(os.path.join(job_dir, "probe.json"), "w") as pf:
            json.dump([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0], pf)
        print(f"[smoke] predict probe published for {job_id}", flush=True)
        return 0

    kind = ARTIFACT_KINDS.get(job_type)
    if kind:
        rel = f"runs/{job_id}/ckpt/{kind}.safetensors"
        full = os.path.join(workdir, rel.replace("/", os.sep))
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "wb") as af:
            af.write(b"fake-" + job_id.encode() + b".safetensors")
        with open(os.path.join(job_dir, "artifacts.json"), "w") as mf:
            json.dump([{"kind": kind, "path": rel}], mf)
        print(f"[{job_type}] artifact {rel} written", flush=True)

    print(f"[done] job {job_id} finished ({job_type})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
