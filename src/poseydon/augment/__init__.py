"""Data augmentation.

An augmentation is a small class, registered by name so ``poseydon list``
can enumerate it, and built from a config entry naming it by ``type:``,
because (unlike features or conditioners) entries carry parameters of their
own -- a probability, and sometimes more.
"""

from poseydon.augment import topology as _topology  # noqa: F401  (registers)
from poseydon.augment.base import AUGMENTATIONS, Augmentation, AugmentPipeline
from poseydon.augment.joint_edit import JointEdit
from poseydon.augment.topology import DropEndEffector, DuplicateJoint

__all__ = [
    "AUGMENTATIONS",
    "AugmentPipeline",
    "Augmentation",
    "DropEndEffector",
    "DuplicateJoint",
    "JointEdit",
]
