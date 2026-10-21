# Integration tests (design §11: service + FakeRunner — CPU, < 5 min)

Every PR. These run the FastAPI app with fakes at every seam
(`storage`, `fs`, `runner/fake`), so the entire SSE path runs on CPU.

Not yet written — the seams (storage/fs/runner) land with the next phase:
auth, chunked upload + prep, tickets, queue admission, SSE, robot predict.
The pure core this suite depends on is complete and green (tests/unit).

Run: `uv run pytest tests/integration -m integration`
