"""Motion tasks: what to do with a trained model."""

from poseydon.ops.base import OPERATIONS, Operation
from poseydon.ops.blend import Blend
from poseydon.ops.generate import Generate
from poseydon.ops.inbetween import Inbetween
from poseydon.ops.retarget import Retarget

__all__ = ["OPERATIONS", "Blend", "Generate", "Inbetween", "Operation", "Retarget"]
