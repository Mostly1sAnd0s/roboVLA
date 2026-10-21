"""Auto-mark tests by directory so `pytest -m "not hardware"` (CI) works
without decorating every test (one place defines the mapping)."""

from __future__ import annotations

from pathlib import Path

import pytest

MARKS = {"unit": "unit", "integration": "integration", "e2e_class": "e2e", "hardware": "hardware"}


def pytest_collection_modifyitems(config, items) -> None:
    root = Path(config.rootpath)
    for item in items:
        try:
            parts = Path(item.fspath).relative_to(root).parts
        except ValueError:
            continue
        if parts and parts[0] in MARKS:
            item.add_marker(getattr(pytest.mark, MARKS[parts[0]]))
