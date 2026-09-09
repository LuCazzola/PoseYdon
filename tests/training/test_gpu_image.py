"""The training image is a claim about hardware; this is where it is checked.

Skipped on any machine without a GPU -- including the CPU test image, which is
where the rest of the suite runs. It is NOT a substitute for the smoke command
in Step 5; it is the regression guard that keeps the compose service and the
Dockerfile from drifting apart.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_the_train_service_declares_the_gpu_and_forwards_wandb():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    train = compose["services"]["train"]
    assert train["devices"] == ["nvidia.com/gpu=all"], "CDI is how this host exposes the GPU"
    assert train["build"]["dockerfile"] == "Dockerfile.train"
    env = train["environment"]
    # Forwarded, never hardcoded: a literal key here would be committed.
    assert env["WANDB_API_KEY"] == "${WANDB_API_KEY:-}"
    assert env["WANDB_ENTITY"] == "${WANDB_ENTITY:-}"


def test_the_train_image_overrides_the_cpu_torch_pin():
    """pyproject pins torch to the CPU index for the test image. If
    Dockerfile.train ever stops overriding that, training silently runs on the
    CPU -- 600k steps that would never finish, with no error to notice.
    """
    dockerfile = (REPO_ROOT / "Dockerfile.train").read_text()
    assert "download.pytorch.org/whl/cu130" in dockerfile
    assert "--reinstall-package torch" in dockerfile


def test_cuda_is_really_usable_when_a_gpu_is_present():
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("no GPU visible (expected in the CPU test image)")
    # Not just is_available(): the wheel's arch list has no sm_121, so the
    # only honest check is that compute actually runs.
    a = torch.randn(512, 512, device="cuda")
    assert torch.isfinite(a @ a).all()
