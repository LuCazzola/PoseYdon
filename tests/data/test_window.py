import numpy as np
import pytest

from poseydon.data.window import WINDOWS, FullClip, RandomCrop, SlidingWindow

RNG = np.random.default_rng(0)


def test_all_policies_registered():
    assert WINDOWS.names() == ["full_clip", "random_crop", "sliding"]


def test_full_clip_takes_everything():
    assert FullClip().count(137) == 1
    assert FullClip().bounds(137, 0, RNG) == (0, 137)


def test_random_crop_stays_inside_the_clip():
    crop = RandomCrop(length=40)
    for _ in range(200):
        start, length = crop.bounds(100, 0, RNG)
        assert length == 40
        assert 0 <= start <= 60


def test_random_crop_yields_short_clips_whole():
    start, length = RandomCrop(length=40).bounds(25, 0, RNG)
    assert (start, length) == (0, 25)


def test_random_crop_can_reach_both_ends():
    crop = RandomCrop(length=10)
    seen = {crop.bounds(12, 0, RNG)[0] for _ in range(500)}
    assert seen == {0, 1, 2}


def test_sliding_covers_the_whole_clip():
    window = SlidingWindow(length=40, stride=20)
    n_frames = 100
    covered = set()
    for i in range(window.count(n_frames)):
        start, length = window.bounds(n_frames, i, RNG)
        covered.update(range(start, start + length))
    assert covered == set(range(n_frames))


def test_sliding_never_runs_past_the_end():
    window = SlidingWindow(length=40, stride=30)
    n_frames = 95
    for i in range(window.count(n_frames)):
        start, length = window.bounds(n_frames, i, RNG)
        assert start + length <= n_frames


def test_sliding_is_deterministic():
    window = SlidingWindow(length=10, stride=5)
    first = [window.bounds(50, i, RNG) for i in range(window.count(50))]
    second = [window.bounds(50, i, np.random.default_rng(99)) for i in range(window.count(50))]
    assert first == second


def test_rejects_bad_configuration():
    with pytest.raises(ValueError, match="length"):
        RandomCrop(length=0)
    with pytest.raises(ValueError, match="stride"):
        SlidingWindow(stride=0)
