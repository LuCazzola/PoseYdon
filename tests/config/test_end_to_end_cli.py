"""ingest -> train -> checkpoint -> sample -> BVH, through the command line."""

import numpy as np
import pytest

from poseydon.cli import main
from poseydon.io.bvh import load_bvh
from tests.ingest.manifest_helper import MANIFEST_DIR


@pytest.fixture(scope="module")
def ingested(truebones_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("cli_corpus")
    assert (
        main(
            [
                "ingest",
                str(truebones_dir),
                "--manifests",
                str(MANIFEST_DIR),
                "--out",
                str(out),
            ]
        )
        == 0
    )
    assert (out / "index.jsonl").is_file()
    return out


def data_overrides(root):
    return [
        f"data.root={root}",
        f"data.index={root}/index.jsonl",
        f"data.manifests={MANIFEST_DIR}",
    ]


TINY = ["model.d_model=32", "model.n_layers=2", "model.ff_size=64"]


def test_ingest_writes_one_clip_per_source(ingested):
    assert len(list((ingested / "aligned").glob("*.npz"))) == 7


def test_train_runs_and_writes_a_checkpoint(ingested, tmp_path):
    out = tmp_path / "runs"
    assert (
        main(
            [
                "train",
                "trainer=debug",
                *TINY,
                "data.batch_size=4",
                "data.window.length=16",
                *data_overrides(ingested),
                "--out",
                str(out),
                "--quiet",
            ]
        )
        == 0
    )
    assert list(out.rglob("*.ckpt")), "training produced no checkpoint"


def test_sample_writes_loadable_bvh(ingested, tmp_path):
    out = tmp_path / "samples"
    assert (
        main(
            [
                "sample",
                *TINY,
                "sampler.steps=5",
                "n_samples=2",
                "n_frames=24",
                "skeleton=Goat",
                *data_overrides(ingested),
                "--out",
                str(out),
            ]
        )
        == 0
    )

    written = sorted(out.glob("*.bvh"))
    assert len(written) == 2
    for path in written:
        anim = load_bvh(path)
        assert anim.n_frames == 24
        assert anim.n_joints == 31  # Goat, including End Sites
        assert np.isfinite(anim.global_positions()).all()


def test_sample_from_a_trained_checkpoint(ingested, tmp_path):
    runs, samples = tmp_path / "runs", tmp_path / "samples"
    main(
        [
            "train",
            "trainer=debug",
            *TINY,
            "data.batch_size=4",
            "data.window.length=16",
            *data_overrides(ingested),
            "--out",
            str(runs),
            "--quiet",
        ]
    )
    checkpoint = next(iter(runs.rglob("*.ckpt")))

    assert (
        main(
            [
                "sample",
                *TINY,
                "sampler.steps=5",
                "n_samples=1",
                "n_frames=16",
                "skeleton=Flamingo",
                *data_overrides(ingested),
                "--checkpoint",
                str(checkpoint),
                "--out",
                str(samples),
            ]
        )
        == 0
    )
    anim = load_bvh(next(iter(samples.glob("*.bvh"))))
    assert anim.n_joints == 40  # Flamingo


def test_sample_rejects_an_unknown_skeleton(ingested, tmp_path):
    assert (
        main(
            [
                "sample",
                *TINY,
                "skeleton=Unicorn",
                *data_overrides(ingested),
                "--out",
                str(tmp_path),
            ]
        )
        == 1
    )
