"""roboVLA image stub for the ``tensorflow_graphics`` package.

The pinned OpenVLA code (openvla/openvla @ c8f03f48) imports
``tensorflow_graphics.geometry.transformation`` at module level
(``prismatic/vla/datasets/rlds/oxe/utils/droid_utils.py``, plus
function-level imports in ``oxe/transforms.py``), but the real package is
uninstallable on this image's Python 3.12: tensorflow-graphics==2021.12.3
(latest release and the official pin) depends on
``tensorflow-addons>=0.10.0``, and tensorflow-addons 0.23.0 (latest) ships
wheels only for cp39-cp311.

The pinned code uses tfg only inside DROID / furniture-bench / iamlab / cmu
dataset transforms -- none of which the roboVLA crosscheck path ('toto', a
pure tf.concat standardization) ever calls. This stub satisfies the import;
every callable raises a loud error if it is ever invoked.
"""

__version__ = "0.0.1+robovla-stub"
