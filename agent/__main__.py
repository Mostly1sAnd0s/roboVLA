"""``python -m agent`` — the runner container's entrypoint (design §10)."""

from .server import main

if __name__ == "__main__":
    main()
