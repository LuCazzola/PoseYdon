from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

_CANDIDATE_DIRS = (
    REPO_ROOT / "tests" / "data" / "truebones",
    REPO_ROOT / "external" / "neural_motion_blending" / "assets" / "truebones",
)

FIXTURE_STEMS = [
    "BrownBear___RiseSwat_132",
    "Coyote___Attack3_224",
    "Crab___Attack3_234",
    "Flamingo_Flamingo_OneLEgBEnt_353",
    "Goat___HeadButt_395",
    "Scorpion___SlowForward_839",
    "Skunk___Spray_891",
]

# Ground truth measured from the files themselves. Joint counts INCLUDE End Sites.
FIXTURE_FACTS = {
    "BrownBear___RiseSwat_132":          {"joints": 38, "frames": 160, "channels": 87},
    "Coyote___Attack3_224":              {"joints": 40, "frames": 91,  "channels": 96},
    "Crab___Attack3_234":                {"joints": 54, "frames": 90,  "channels": 135},
    "Flamingo_Flamingo_OneLEgBEnt_353":  {"joints": 40, "frames": 201, "channels": 87},
    "Goat___HeadButt_395":               {"joints": 31, "frames": 79,  "channels": 72},
    "Scorpion___SlowForward_839":        {"joints": 63, "frames": 80,  "channels": 153},
    "Skunk___Spray_891":                 {"joints": 35, "frames": 121, "channels": 75},
}


def fixture_dir() -> Path | None:
    """First directory holding a complete set of the seven fixture BVHs, else None."""
    for directory in _CANDIDATE_DIRS:
        if directory.is_dir() and all(
            (directory / f"{stem}.bvh").is_file() for stem in FIXTURE_STEMS
        ):
            return directory
    return None


@pytest.fixture(scope="session")
def truebones_dir() -> Path:
    directory = fixture_dir()
    if directory is None:
        pytest.skip(
            "Truebones BVH fixtures not found. Expected a complete set in either "
            f"{_CANDIDATE_DIRS[0]} or {_CANDIDATE_DIRS[1]}."
        )
    return directory


@pytest.fixture(params=FIXTURE_STEMS)
def bvh_fixture(request, truebones_dir) -> Path:
    """Parameterized: every test using this runs once per fixture file."""
    return truebones_dir / f"{request.param}.bvh"


@pytest.fixture(scope="session")
def fixture_facts() -> dict[str, dict[str, int]]:
    """Ground-truth counts, exposed as a fixture.

    Tests in subdirectories must NOT do ``from conftest import ...`` -- whether
    ``tests/`` lands on ``sys.path`` depends on pytest's import mode. Fixtures
    are resolved up the directory tree and always work.
    """
    return FIXTURE_FACTS


@pytest.fixture(scope="session")
def fixture_stems() -> list[str]:
    return list(FIXTURE_STEMS)
