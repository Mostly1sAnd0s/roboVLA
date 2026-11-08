"""The SmolVLA trackio sink (images/trackio_sink.py) — the local stand-in
for the Trackio surface lerobot v0.7.0's TrackioLogger uses.

lerobot-train logs every --log_freq loop steps through
TrackioLogger.log_dict -> trackio.log({"train/<key>": v, ...}, step=...).
The wrapper shadows the real package with this module (image path
/opt/robovla/sink/trackio.py, first on the child's PYTHONPATH): the curve
lands in the job's standard metrics.jsonl (one metrics channel — the
agent poll / SSE / sparkline consume it, no change).

Pinned here: the row mapping (lerobot MetricsTracker keys -> house keys,
the eta/vram derivations, eval passthrough, non-numeric drop), the env
contract (shared NAMES with the crosscheck wandb sink — one vocabulary),
the no-op default, the fail-loud edges (unknown attributes, non-int
step), the artifact no-ops (loud-never-crash), the init/finish lines
never entering metrics.jsonl, name independence (the aliased image
deploy), and the Dockerfile-dest / wrapper-SINK_DIR agreement.
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

from images import trackio_sink, wandb_sink  # noqa: E402

SINK_FILE = REPO / "images" / "trackio_sink.py"

#: One lerobot MetricsTracker.to_dict() window (v0.7.0 key set, prefixed
#: by TrackioLogger.log_dict).
LEROBOT_PAYLOAD = {
    "train/steps": 42, "train/samples": 336, "train/episodes": 4,
    "train/epochs": 1.2, "train/loss": 0.4172, "train/grad_norm": 1.9,
    "train/lr": 9.9e-5, "train/dataloading_s": 0.01,
    "train/preprocessing_s": 0.03, "train/update_s": 0.51,
    "train/step_s": 0.55, "train/samples_per_s": 14.5,
    "train/gpu_mem_gb": 6.34,
}


def _set_env(monkeypatch, sink: pathlib.Path, total: str | None = "210"):
    monkeypatch.setenv(trackio_sink.ENV_SINK, str(sink))
    if total is not None:
        monkeypatch.setenv(trackio_sink.ENV_TOTAL, total)


def _unset_env(monkeypatch):
    monkeypatch.delenv(trackio_sink.ENV_SINK, raising=False)
    monkeypatch.delenv(trackio_sink.ENV_TOTAL, raising=False)


def _rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(ln) for ln in
            path.read_text().splitlines() if ln.strip()]


# -- env vocabulary (one contract across both sinks) --------------------------------


def test_env_names_shared_with_the_wandb_sink():
    assert trackio_sink.ENV_SINK == wandb_sink.ENV_SINK
    assert trackio_sink.ENV_TOTAL == wandb_sink.ENV_TOTAL


# -- row mapping ---------------------------------------------------------------------


def test_row_mapping_house_keys(tmp_path, monkeypatch):
    sink = tmp_path / "metrics.jsonl"
    _set_env(monkeypatch, sink)
    trackio_sink.log(LEROBOT_PAYLOAD, step=42)
    row = _rows(sink)[0]
    assert row["step"] == 42
    assert row["total"] == 210
    assert row["loss"] == 0.4172
    assert row["lr"] == 9.9e-5
    assert row["grad_norm"] == 1.9
    assert row["vram_mb"] == pytest.approx(6.34 * 1024)
    # eta = remaining loop steps x window-average step time
    assert row["eta_s"] == int((210 - 42) * 0.55)
    # passthrough with the prefix stripped (student-visible extras)
    assert row["samples"] == 336
    assert row["samples_per_s"] == 14.5


def test_eval_prefix_passes_through(tmp_path, monkeypatch):
    sink = tmp_path / "metrics.jsonl"
    _set_env(monkeypatch, sink)
    trackio_sink.log({"eval/eval_loss": 0.9}, step=100)
    assert _rows(sink)[0]["eval_loss"] == 0.9


def test_non_numeric_and_unknown_modes_dropped(tmp_path, monkeypatch):
    sink = tmp_path / "metrics.jsonl"
    _set_env(monkeypatch, sink)
    trackio_sink.log({"train/loss": 0.5, "train/text": "x",
                      "train/nan": float("nan"), "other/k": 1.0}, step=1)
    row = _rows(sink)[0]
    assert row["loss"] == 0.5
    assert "text" not in row and "nan" not in row and "k" not in row


def test_eta_absent_without_total_or_future_step(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path / "m1.jsonl", total=None)
    trackio_sink.log({"train/loss": 1.0, "train/step_s": 0.5}, step=1)
    assert "eta_s" not in _rows(tmp_path / "m1.jsonl")[0]


# -- no-op default + fail-loud edges ---------------------------------------------------


def test_no_sink_env_is_a_silent_noop(tmp_path, monkeypatch):
    _unset_env(monkeypatch)
    trackio_sink.log(LEROBOT_PAYLOAD, step=1)  # must not raise or write
    assert not (tmp_path / "metrics.jsonl").exists()


@pytest.mark.parametrize("step", [None, "3", 1.0, True])
def test_step_must_be_int(step):
    with pytest.raises(ValueError):
        trackio_sink.log({"train/loss": 1.0}, step=step)


def test_unknown_module_attribute_fails_loud():
    with pytest.raises(AttributeError) as ei:
        trackio_sink.watch
    assert "trackio_sink" in str(ei.value)


def test_run_marker_url_local_and_unknown_attrs_fail():
    run = trackio_sink.init(project="p", name="n")
    assert run.url is None
    with pytest.raises(AttributeError):
        run.log


# -- artifacts: loud, never crash ------------------------------------------------------


def test_artifact_calls_warn_once_per_point_and_never_raise(tmp_path,
                                                            monkeypatch,
                                                            capsys):
    trackio_sink._warned.clear()
    try:
        art = trackio_sink.Artifact("policy", type="model")
        art.add_file("x.safetensors")
        trackio_sink.log_artifact(art)
        trackio_sink.log_artifact(art)  # second call: no second warning
    finally:
        trackio_sink._warned.clear()
    out = capsys.readouterr().out
    # one line per DISTINCT surface point (add_file + log_artifact), the
    # repeated log_artifact contributes nothing
    assert out.count("disable_artifact") == 2


# -- init/finish never touch metrics.jsonl ---------------------------------------------


def test_init_and_finish_only_print(tmp_path, monkeypatch, capsys):
    sink = tmp_path / "metrics.jsonl"
    _set_env(monkeypatch, sink)
    trackio_sink.init(project="robovla-smolvla", name="j_1")
    trackio_sink.finish()
    assert not sink.exists()
    out = capsys.readouterr().out
    assert "init" in out and "finish" in out


# -- the aliased image deploy (name independence + dest agreement) ----------------------


def test_loads_under_an_aliased_name(tmp_path, monkeypatch):
    # the image ships this file AS trackio.py: behavior must not depend on
    # the module's real name (the wandb-sink rule).
    spec = importlib.util.spec_from_file_location("trackio", SINK_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sink = tmp_path / "metrics.jsonl"
    monkeypatch.setenv(mod.ENV_SINK, str(sink))
    monkeypatch.setenv(mod.ENV_TOTAL, "10")
    mod.log({"train/loss": 0.25}, step=5)
    assert _rows(sink)[0]["loss"] == 0.25


def test_dockerfile_dest_matches_the_wrapper_sink_dir():
    from images import smolvla_finetune
    docker = (REPO / "images" / "Dockerfile.cuda-smolvla").read_text()
    assert ("COPY images/trackio_sink.py "
            f"{smolvla_finetune.SINK_DIR}/trackio.py") in docker
