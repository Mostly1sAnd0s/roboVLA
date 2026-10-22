"""images/tfg_stub — the image-only tensorflow_graphics stub.

The pinned OpenVLA code (openvla/openvla @ c8f03f48) imports
``tensorflow_graphics.geometry.transformation`` at module level
(``prismatic/vla/datasets/rlds/oxe/utils/droid_utils.py``), but the real
package is uninstallable on this image's Python 3.12:
tensorflow-graphics==2021.12.3 (latest release and the official pin) depends
on ``tensorflow-addons>=0.10.0`` and tensorflow-addons 0.23.0 (latest) ships
wheels only for cp39-cp311. The pinned code uses tfg only inside DROID /
furniture-bench / iamlab / cmu dataset transforms — none of which the
crosscheck path ('toto', a pure tf.concat standardization) ever calls. The
stub satisfies the import; every callable raises a loud error on invocation
(a stub that silently returns wrong math would be worse than a crash).

CPU-pinned here because the stub is plain Python; the image build installs
it (``uv pip install /opt/robovla/images/tfg_stub``).
"""

from __future__ import annotations

import sys
import types

import pytest

# The stub's import root is images/tfg_stub (where the tensorflow_graphics/
# package lives); the image installs it via pip, the test imports it from
# the tree.
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
STUB_ROOT = REPO / "images" / "tfg_stub"

LEAF_FNS = [
    ("euler", "from_quaternion"),          # oxe/transforms.py (furniture_bench, iamlab, cmu)
    ("euler", "from_rotation_matrix"),     # oxe/utils/droid_utils.py (rmat_to_euler)
    ("rotation_matrix_3d", "from_euler"),  # droid_utils (euler_to_rmat)
    ("rotation_matrix_3d", "inverse"),     # droid_utils (invert_rmat)
]


@pytest.fixture(scope="module")
def tfg() -> "types.ModuleType":
    """The import the pinned openvla code performs:
    ``import tensorflow_graphics.geometry.transformation as tfg``.

    Generator fixture: the path/sys.modules cleanup must run at teardown,
    not during setup (a plain-function fixture's ``finally`` would fire the
    moment it returns, before any test runs).
    """
    sys.path.insert(0, str(STUB_ROOT))
    try:
        import tensorflow_graphics.geometry.transformation as tfg_mod

        yield tfg_mod
    finally:
        sys.path.remove(str(STUB_ROOT))
        for name in list(sys.modules):
            if name == "tensorflow_graphics" or name.startswith("tensorflow_graphics."):
                del sys.modules[name]


def test_stub_version_marker(tfg):
    import tensorflow_graphics

    assert "robovla-stub" in tensorflow_graphics.__version__


@pytest.mark.parametrize("submodule,fn", LEAF_FNS)
def test_leaf_functions_raise_loudly(tfg, submodule, fn):
    target = getattr(tfg, submodule)
    leaf = getattr(target, fn)
    assert callable(leaf)
    with pytest.raises(RuntimeError) as ei:
        leaf(None)
    # The message must name the reason (unresolvable on 3.12) and the fix
    # boundary (DROID/other-embodiment transforms must not run here).
    msg = str(ei.value)
    assert "tensorflow-addons" in msg
    assert "3.12" in msg
