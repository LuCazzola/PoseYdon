"""Motion tasks: what to do with a trained model."""

from poseydon.ops.base import OPERATIONS, Operation
from poseydon.ops.blend import Blend
from poseydon.ops.generate import Generate
from poseydon.ops.inbetween import Inbetween

__all__ = ["OPERATIONS", "Blend", "Generate", "Inbetween", "Operation"]
