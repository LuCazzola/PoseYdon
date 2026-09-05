"""Data augmentation.

An augmentation is a small class, registered by name so ``poseydon list``
can enumerate it, and instantiated by full ``_target_`` path from config
because (unlike features or conditioners) entries carry parameters of their
own -- a probability, and sometimes more. See the design spec for why this
mirrors ``conditioners:`` in philosophy but not in config shape.
"""

from poseydon.augment import topology as _topology  # noqa: F401  (registers)
from poseydon.augment.base import AUGMENTATIONS, Augmentation, AugmentPipeline
from poseydon.augment.joint_edit import JointEdit
from poseydon.augment.topology import DropEndEffector, DuplicateJoint

__all__ = [
    "AUGMENTATIONS",
    "Augmentation",
    "AugmentPipeline",
    "JointEdit",
    "DropEndEffector",
    "DuplicateJoint",
]
