"""Auto-mark tests by layer directory so ``pytest -m "not hardware"`` (CI)
works without decorating every test (one place defines the mapping).

Layout: tests/{unit,integration,e2e_class,hardware}/... — the layer is the
directory directly under tests/. (Earlier version matched on parts[0], which
is always ``tests`` and silently marked nothing — the hardware gate would
have collected zero tests.)
"""

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
        # tests/<layer>/file.py -> layer at index 1
        if len(parts) >= 3 and parts[0] == "tests" and parts[1] in MARKS:
            item.add_marker(getattr(pytest.mark, MARKS[parts[1]]))
