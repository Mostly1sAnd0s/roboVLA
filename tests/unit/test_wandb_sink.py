"""The crosscheck wandb sink (images/wandb_sink.py) — the local stand-in for
the wandb API surface the pinned official finetune.py uses.

The pinned script (openvla SHA c8f03f48, vla-scripts/finetune.py, read
2026-10-23) touches wandb exactly twice, main process only:

    wandb.init(entity=..., project=..., name=...)
    wandb.log({"train_loss": ..., "action_accuracy": ..., "l1_loss": ...},
              step=gradient_step_idx)          # every 10 gradient steps

Under WANDB_MODE=disabled (no credentials on the LAN) those calls were no-ops
and the official's per-step loss was lost — the documented crosscheck
degradation (docs/runner-image.md). The wrapper now shadows the real wandb
package with this module (image path /opt/robovla/sink/wandb.py, first on the
child's PYTHONPATH): the sink maps the official payload into the job's
standard metrics.jsonl, so the agent's incremental poll / SSE metrics / the
run-detail sparkline consume it with no change (one metrics channel, §11 DRY).

Pinned here: the row mapping (house key set + finite-numeric passthrough),
the env contract (target/total/lr — read per call, no import-time caching),
the no-op default (sink env unset = today's disabled behavior), the
fail-loud edges (unknown attributes, non-int step — the pinned surface is
init/log, anything else means a re-pin), the init line going to the job log
(never into metrics.jsonl — the agent treats every JSON dict line as a
metrics row), and the name independence the image's aliased deploy depends
on (the same file must behave identically loaded as a foreign module name).
"""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys

import pytest

REPO = pathlib.Path(__file__).parents[2]
sys.path.insert(0, str(REPO))

from images import wandb_sink  # noqa: E402

SINK_FILE = REPO / "images" / "wandb_sink.py"
#: The official payload shape (pinned script, main process, every 10 steps).
OFFICIAL_PAYLOAD = {"train_loss": 12.17, "action_accuracy": 0.03,
                    "l1_loss": 0.98}


def _set_env(monkeypatch, sink: pathlib.Path, total: str | None = "54",
             lr: str | None = "0.0005") -> None:
    monkeypatch.setenv(wandb_sink.ENV_SINK, str(sink))
    if total is not None:
        monkeypatch.setenv(wandb_sink.ENV_TOTAL, total)
    if lr is not None:
        monkeypatch.setenv(wandb_sink.ENV_LR, lr)


def _unset_env(monkeypatch) -> None:
    for name in (wandb_sink.ENV_SINK, wandb_sink.ENV_TOTAL, wandb_sink.ENV_LR):
        monkeypatch.delenv(name, raising=False)


def _rows(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in
            path.read_text().splitlines() if line.strip()]


# -- init: the job-log line, never a metrics row ---------------------------------


def test_init_announces_to_the_job_log(tmp_path, monkeypatch, capsys):
    _unset_env(monkeypatch)
    run = wandb_sink.init(entity="local", project="robovla-crosscheck",
                          name="ft+openvla-7b+toto--j_1")
    assert run is not None  # the pinned script ignores the return value
    out = capsys.readouterr().out
    assert "robovla-crosscheck" in out and "ft+openvla-7b+toto--j_1" in out
    assert not (tmp_path / "metrics.jsonl").exists()


def test_init_writes_nothing_to_the_sink_even_when_configured(
        tmp_path, monkeypatch):
    """The agent treats every JSON dict line in metrics.jsonl as a metrics
    row — an init header line would corrupt the stream (step/total None)."""
    sink = tmp_path / "metrics.jsonl"
    _set_env(monkeypatch, sink)
    wandb_sink.init(entity="local", project="p", name="n")
    assert not sink.exists()


# -- log: the row mapping ---------------------------------------------------------


def test_log_maps_the_official_payload_to_the_house_row(tmp_path, monkeypatch):
    sink = tmp_path / "metrics.jsonl"
    _set_env(monkeypatch, sink)
    wandb_sink.log(dict(OFFICIAL_PAYLOAD), step=10)
    rows = _rows(sink)
    assert rows == [{"step": 10, "total": 54, "loss": 12.17, "lr": 0.0005,
                     "action_accuracy": 0.03, "l1_loss": 0.98}]
    # train_loss is MAPPED to the house key, not duplicated under its name
    assert "train_loss" not in rows[0]


def test_log_appends_in_order(tmp_path, monkeypatch):
    sink = tmp_path / "metrics.jsonl"
    _set_env(monkeypatch, sink)
    wandb_sink.log(dict(OFFICIAL_PAYLOAD), step=0)
    wandb_sink.log({"train_loss": 2.2}, step=10)
    assert [r["step"] for r in _rows(sink)] == [0, 10]
    assert [r["loss"] for r in _rows(sink)] == [12.17, 2.2]


def test_missing_total_and_lr_envs_omit_the_keys(tmp_path, monkeypatch):
    sink = tmp_path / "metrics.jsonl"
    _set_env(monkeypatch, sink, total=None, lr=None)
    wandb_sink.log(dict(OFFICIAL_PAYLOAD), step=30)
    row = _rows(sink)[0]
    assert row == {"step": 30, "loss": 12.17, "action_accuracy": 0.03,
                   "l1_loss": 0.98}


def test_non_numeric_and_non_finite_values_are_dropped(tmp_path, monkeypatch):
    sink = tmp_path / "metrics.jsonl"
    _set_env(monkeypatch, sink)
    wandb_sink.log({"train_loss": 1.0, "action_accuracy": float("nan"),
                    "note": "hello", "flag": True, "l1_loss": 2}, step=40)
    row = _rows(sink)[0]
    assert row == {"step": 40, "total": 54, "loss": 1.0, "lr": 0.0005,
                   "l1_loss": 2}


def test_log_without_sink_env_is_a_noop(tmp_path, monkeypatch):
    """Unset sink env = today's WANDB_MODE=disabled behavior (no error)."""
    _unset_env(monkeypatch)
    wandb_sink.log(dict(OFFICIAL_PAYLOAD), step=10)  # must not raise
    assert not (tmp_path / "metrics.jsonl").exists()


def test_env_is_read_per_call_not_at_import(tmp_path, monkeypatch):
    """The child's env is set at spawn, after the module import — no
    import-time caching of the sink path."""
    _unset_env(monkeypatch)
    assert wandb_sink._sink_path() is None
    sink = tmp_path / "metrics.jsonl"
    monkeypatch.setenv(wandb_sink.ENV_SINK, str(sink))
    wandb_sink.log({"train_loss": 1.5}, step=0)
    assert _rows(sink)[0]["loss"] == 1.5


# -- fail-loud edges (the pinned surface is init/log) ------------------------------


def test_step_must_be_an_int(tmp_path, monkeypatch):
    _set_env(monkeypatch, tmp_path / "metrics.jsonl")
    for bad in ("10", 10.0, True, None):
        with pytest.raises(ValueError):
            wandb_sink.log({"train_loss": 1.0}, step=bad)
    with pytest.raises(ValueError):  # the pinned script always passes step
        wandb_sink.log({"train_loss": 1.0})


def test_unknown_module_attributes_fail_loud():
    with pytest.raises(AttributeError) as ei:
        wandb_sink.login()
    assert "wandb sink" in str(ei.value)
    with pytest.raises(AttributeError):
        wandb_sink.run  # the pinned script never uses wandb.run
    assert isinstance(wandb_sink.__version__, str)


# -- name independence (the image ships the file as /opt/robovla/sink/wandb.py) ---


def test_aliased_module_behaves_identically(tmp_path, monkeypatch):
    """The Dockerfile COPYs the file under the aliased name wandb.py; the
    module must carry no self-name assumption (load it as a foreign module
    and pin the same contract)."""
    spec = importlib.util.spec_from_file_location("wandb_under_test", SINK_FILE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sink = tmp_path / "metrics.jsonl"
    _set_env(monkeypatch, sink)
    mod.log(dict(OFFICIAL_PAYLOAD), step=10)
    assert _rows(sink)[0]["loss"] == 12.17


# -- image wiring (the CPU pin for the Dockerfile; the hardware row verifies) ------


def test_dockerfile_cuda_bakes_the_sink_at_the_aliased_path():
    """Wrapper SINK_DIR (the child's PYTHONPATH) and the Dockerfile dest must
    agree — a drift means `import wandb` in the crosscheck child resolves to
    the real package again (silent loss of the curve)."""
    from images import crosscheck_finetune as xf
    assert xf.SINK_DIR == "/opt/robovla/sink"
    df = (REPO / "images" / "Dockerfile.cuda").read_text()
    assert f"COPY images/wandb_sink.py {xf.SINK_DIR}/wandb.py" in df
