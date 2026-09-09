"""The training image is a claim about hardware; this is where it is checked.

Skipped on any machine without a GPU -- including the CPU test image, which is
where the rest of the suite runs. It is NOT a substitute for the smoke command
in Step 5; it is the regression guard that keeps the compose service and the
Dockerfile's `train` stage from drifting apart.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _stage_of(dockerfile: Path, name: str) -> str:
    """One named stage's text, from its own FROM up to the next one.

    Substring checks against the whole file would pass on a mention in another
    stage's comment, which is not the same claim at all.
    """
    text = dockerfile.read_text()
    starts = [(m.start(), m.group(1)) for m in re.finditer(r"^FROM .+ AS (\w+)$", text, re.MULTILINE)]
    for index, (offset, stage) in enumerate(starts):
        if stage == name:
            end = starts[index + 1][0] if index + 1 < len(starts) else len(text)
            return text[offset:end]
    raise AssertionError(f"no stage named {name!r} in {dockerfile}")


def test_the_train_service_declares_the_gpu_and_forwards_wandb():
    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    train = compose["services"]["train"]
    assert train["devices"] == ["nvidia.com/gpu=all"], "CDI is how this host exposes the GPU"
    # One Dockerfile, four named stages; each service picks its own.
    assert train["build"]["target"] == "train"
    assert "dockerfile" not in train["build"], "the per-image Dockerfiles were consolidated"
    env = train["environment"]
    # Forwarded, never hardcoded: a literal key here would be committed.
    assert env["WANDB_API_KEY"] == "${WANDB_API_KEY:-}"
    assert env["WANDB_ENTITY"] == "${WANDB_ENTITY:-}"


def test_the_train_stage_overrides_the_cpu_torch_pin():
    """pyproject pins torch to the CPU index for the test image. If the `train`
    stage ever stops overriding that, training silently runs on the CPU --
    600k steps that would never finish, with no error to notice.

    Read from the `train` STAGE specifically, not from the whole file: a match
    anywhere in a four-stage Dockerfile would also be satisfied by a stray
    mention in a comment under `compat`.
    """
    stage = _stage_of(REPO_ROOT / "Dockerfile", "train")
    assert "download.pytorch.org/whl/cu130" in stage
    assert "--reinstall-package torch" in stage


def test_every_compose_service_targets_a_stage_that_exists():
    """The services and the stages are now two halves of one file; a typo in
    either silently builds the wrong image, or fails only at `docker build`.
    """
    dockerfile = (REPO_ROOT / "Dockerfile").read_text()
    stages = set(re.findall(r"^FROM .+ AS (\w+)$", dockerfile, re.MULTILINE))
    assert stages == {"base", "test", "train", "compat", "fbx"}

    compose = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
    for name, service in compose["services"].items():
        target = service.get("build", {}).get("target")
        assert target in stages, f"service `{name}` targets unknown stage {target!r}"


def test_the_fbx_stage_still_carries_its_system_python_packages():
    """The fbx stage shares nothing with `base` -- Blender links against the
    SYSTEM Python, so its dependencies are apt packages, not the venv.
    Consolidating the Dockerfiles is exactly the kind of change that would
    quietly reparent it onto `base` and break every FBX script.
    """
    stage = _stage_of(REPO_ROOT / "Dockerfile", "fbx")
    assert "FROM debian:bookworm-slim" in stage, "fbx must not be based on `base`"
    for package in ("blender", "python3-yaml", "python3-numpy", "python3-scipy", "python3-pytest"):
        assert package in stage


def test_cuda_is_really_usable_when_a_gpu_is_present():
    torch = pytest.importorskip("torch")

    if not torch.cuda.is_available():
        pytest.skip("no GPU visible (expected in the CPU test image)")
    # Not just is_available(): the wheel's arch list has no sm_121, so the
    # only honest check is that compute actually runs.
    a = torch.randn(512, 512, device="cuda")
    assert torch.isfinite(a @ a).all()
