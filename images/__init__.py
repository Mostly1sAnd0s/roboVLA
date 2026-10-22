"""images/ — the runner-image package (design §6, §10).

Baked into the runner image (NOT part of the console wheel):
  contract.py            the job file contract (pure; shared by job + engine)
  robovla_job.py         the job program (smoke/train/validate/merge/selftests)
  engine.py              the QLoRA engine (lazy GPU-stack boundary)
  engine_jobs.py         the job dispatch behind robovla_job (the <=500 split)
  vla_compat.py          the vendored official prompt builder (pinned SHA)
  rlds_writer.py         the crosscheck subset -> official RLDS/TFDS dataset
  crosscheck_finetune.py the crosscheck wrapper (official finetune.py)
  tfg_stub/              the tensorflow_graphics stub (unresolvable on py3.12;
                         image installs it; pinned code only touches it in
                         DROID-only transforms the crosscheck never runs)
  Dockerfile.cuda        the CUDA image (Blackwell, cu128)
  Dockerfile.rocm        the ROCm image (home dev lab, gfx1151)

The image's /opt/robovla/bin entry scripts put /opt/robovla on sys.path and
import from here — one package, both on the dev host and in the container.
"""
