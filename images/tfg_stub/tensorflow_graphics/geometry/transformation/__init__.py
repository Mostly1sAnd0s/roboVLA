"""Stub of ``tensorflow_graphics.geometry.transformation``.

The pinned openvla code imports this package (``as tfg`` / ``as tft``) and
uses only the leaf functions below. ``euler`` and ``rotation_matrix_3d`` are
real submodules so attribute access (``tfg.euler.from_quaternion``) behaves
exactly like the real package; every leaf raises on call.
"""

from . import euler, rotation_matrix_3d  # noqa: F401

__all__ = ["euler", "rotation_matrix_3d"]
