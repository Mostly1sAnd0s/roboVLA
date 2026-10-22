"""images/finetune_patch.diff — the build-time workaround for the pinned
official finetune.py (host65 attempt 6, 2026-10-22).

The pinned openvla finetune.py (SHA c8f03f48) cannot complete even ONE
QLoRA training step on this image's probe-verified stack (torch 2.9.0 /
transformers 4.51.3): its hardcoded DDP wrap (no ``static_graph``) meets
the gradient checkpointing that ``prepare_model_for_kbit_training``
enables, and torch 2.9's implicit NON-reentrant checkpoint default makes
the first backward raise "Expected to mark a variable ready only once"
(OpenVLA issue #90 — same parameter index 875, same stack class — with
the maintainer-acknowledged fix: ``static_graph=True`` + explicit
reentrant checkpointing). No argv / env reaches the DDP args (hardcoded),
and the crosscheck design pins the effective batch (1 x 8), so the only
way to run the official engine is a MINIMAL, hash-pinned, build-time
patch — the exact #90 workaround, mathematically neutral (a DDP
communication hint at world_size 1 / a recomputation implementation
detail).

The CPU suite pins the PATCH ARTIFACT + its Dockerfile wiring (the pinned
openvla repo is fetched at image build, so the apply itself is verified
there: ``git apply --check`` first — the build fails if the pinned file
ever drifts from the patch context — then a sha256 pin of the patched
file).
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).parents[2]
PATCH = REPO / "images" / "finetune_patch.diff"
DOCKERFILE = REPO / "images" / "Dockerfile.cuda"


def test_patch_is_well_formed_single_file_diff():
    text = PATCH.read_text()
    assert text.startswith("--- a/vla-scripts/finetune.py\n")
    assert "+++ b/vla-scripts/finetune.py\n" in text
    # one hunk, one file touched — nothing else in the official tree
    assert sum(1 for line in text.splitlines()
               if line.startswith("@@")) == 1
    changed = {line for line in text.splitlines()
               if line.startswith(("--- a/", "+++ b/"))}
    assert changed == {"--- a/vla-scripts/finetune.py",
                       "+++ b/vla-scripts/finetune.py"}


def test_patch_is_the_issue90_workaround():
    text = PATCH.read_text()
    # part 1: the DDP line gains static_graph
    assert ("+    vla = DDP(vla, device_ids=[device_id], "
            "find_unused_parameters=True, gradient_as_bucket_view=True, "
            "static_graph=True)") in text
    # part 2: checkpointing re-enabled EXPLICITLY reentrant
    assert "+    vla.module.gradient_checkpointing_enable(" in text
    assert '+        gradient_checkpointing_kwargs={"use_reentrant": True})' \
        in text
    # the minus set is EXACTLY the original DDP line (no other official
    # code is touched)
    minus = [line for line in text.splitlines()
             if line.startswith("-") and not line.startswith("---")]
    assert len(minus) == 1
    assert minus[0].startswith("-    vla = DDP(vla, device_ids=[device_id]")
    assert "static_graph" not in minus[0]


def test_dockerfile_applies_the_patch_after_sha_checkout():
    text = DOCKERFILE.read_text()
    assert "COPY images/finetune_patch.diff /tmp/finetune_patch.diff" in text
    # --check FIRST: if the pinned file drifts from the patch context the
    # build fails (lockstep with the SHA, not a silent fuzzy apply)
    assert "git -C /opt/openvla apply --check /tmp/finetune_patch.diff" in text
    assert "git -C /opt/openvla apply /tmp/finetune_patch.diff" in text
    # the PATCHED file is hash-pinned (drift = build failure)
    assert re.search(r"echo \"[0-9a-f]{64}  /opt/openvla/vla-scripts/"
                     r"finetune\.py\" \| sha256sum -c", text), \
        "the patched finetune.py must be sha256-pinned in the Dockerfile"
    # ordering: the apply layer sits AFTER the SHA fetch
    fetch = text.index("git -C /opt/openvla fetch")
    apply_check = text.index("apply --check /tmp/finetune_patch.diff")
    assert fetch < apply_check
