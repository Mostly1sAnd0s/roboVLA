"""images/ — the runner-image package (design §6, §10).

Baked into the runner image (NOT part of the console wheel):
  contract.py            the job file contract (pure; shared by job + engine)
  robovla_job.py         the job program (smoke/train/validate/merge/selftests)
  engine.py              the QLoRA engine (lazy GPU-stack boundary)
  crosscheck_finetune.py the crosscheck wrapper (official finetune.py)
  Dockerfile.cuda        the CUDA image (Blackwell, cu128)
  Dockerfile.rocm        the ROCm image (home dev lab, gfx1151)

The image's /opt/robovla/bin entry scripts put /opt/robovla on sys.path and
import from here — one package, both on the dev host and in the container.
"""
