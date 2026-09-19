from __future__ import annotations

import pytest

from poseydon.core.spec import FeatureSpec


@pytest.fixture
def spec_stub() -> FeatureSpec:
    """One narrow block: these tests care about joint indexing, not layout."""
    return FeatureSpec(blocks=(("ric_pos", 1),))
