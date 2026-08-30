"""In-betweening: generate the middle, keep the ends."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from poseydon.core.batch import Cond
from poseydon.ops.base import OPERATIONS, Operation
from poseydon.process.gaussian import GaussianDiffusion
from poseydon.sampling.base import Control
from poseydon.sampling.controls.inbetween import Inbetween as InbetweenControl


@OPERATIONS.register("inbetween")
class Inbetween(Operation):
    """Fill a gap between known frames.

    Nothing about the model changes: the known frames are imputed at every
    sampling step, so any denoiser trained for synthesis can do this.
    """

    def __init__(
        self, reference: torch.Tensor, known: torch.Tensor, process: GaussianDiffusion
    ) -> None:
        self.reference = reference
        self.known = known
        self.process = process

    @classmethod
    def from_endpoints(
        cls,
        reference: torch.Tensor,
        process: GaussianDiffusion,
        head: int,
        tail: int,
    ) -> Inbetween:
        """Keep the first ``head`` and last ``tail`` frames, generate the rest."""
        n_frames = reference.shape[-1]
        if head + tail >= n_frames:
            raise ValueError(
                f"keeping {head} + {tail} frames leaves nothing to generate in "
                f"a {n_frames}-frame window"
            )
        known = torch.zeros(reference.shape[0], n_frames, dtype=torch.bool)
        known[:, :head] = True
        known[:, n_frames - tail :] = True
        return cls(reference=reference, known=known, process=process)

    def build_controls(self, cond: Cond) -> Sequence[Control]:
        return [
            InbetweenControl(
                reference=self.reference, known=self.known, process=self.process
            )
        ]
