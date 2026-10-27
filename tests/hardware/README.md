# Hardware-gated tests (design section 11: GPU_TESTS=1)

Manual. Run at node join / stack changes. Never in CI:
- `gpu_selftest` per node (via the agent, in-container): torch/CUDA visible.
  Load 7.5B. 8-bit fwd/bwd. bf16 fwd/bwd where it fits. predict_action on
  held-out images. 10-step QLoRA.
- `home_selftest` (self-contained dev lab, ROCm gfx1151).
- Robot (Pi): token auth. LAN predict round-trip. Stale-action watchdog.
  Slot lease expiry (per class milestone M5-M7).

Run: `GPU_TESTS=1 uv run pytest tests/hardware -m hardware`
