"""Composable loss terms."""

from poseydon.losses import footskate as _footskate  # noqa: F401  (registers terms)
from poseydon.losses import geodesic as _geodesic  # noqa: F401
from poseydon.losses import kl as _kl  # noqa: F401
from poseydon.losses import simple as _simple  # noqa: F401
from poseydon.losses.base import LOSSES, NORM_STATS, LossTerm

__all__ = ["LOSSES", "NORM_STATS", "LossTerm"]
