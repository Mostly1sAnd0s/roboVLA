"""Stub of ``tensorflow_graphics.geometry.transformation.euler``.

Leaf functions raise a loud ``RuntimeError`` on call (the DROID /
furniture-bench / iamlab / cmu transforms that would invoke them never run
on the crosscheck path -- a stub that silently returned wrong math would be
worse than a crash).
"""


def _unavailable(*_args, **_kwargs):
    raise RuntimeError(
        "tensorflow_graphics is a roboVLA image stub: the real package "
        "(tensorflow-graphics==2021.12.3) is uninstallable on Python 3.12 "
        "(its tensorflow-addons dependency ships no cp312 wheel). The "
        "DROID / other-embodiment transforms that call this must not run "
        "on the crosscheck path."
    )


from_rotation_matrix = _unavailable
from_quaternion = _unavailable
