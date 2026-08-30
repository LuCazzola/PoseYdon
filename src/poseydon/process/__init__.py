"""Generative processes: corruption, parameterization, and their inverses."""

from poseydon.process.base import PROCESSES, Process
from poseydon.process.flow import FlowMatching
from poseydon.process.gaussian import GaussianDiffusion

__all__ = ["PROCESSES", "FlowMatching", "GaussianDiffusion", "Process"]
