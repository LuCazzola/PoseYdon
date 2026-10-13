# Testing: what is covered, and how to trust it

As of 2026-10-12 the suite is two files. Everything else was deleted because
it had grown too large to troubleshoot with.

| file | covers |
|---|---|
| `tests/test_roundtrip.py` | the [data round trip](../concepts/data/roundtrip.md), on every prepared clip of one dataset, in BVH (`test_round_trip`) and FBX (`test_round_trip_fbx`), plus `test_up_axis`: one rig turned to Z-up and declared so |
| `tests/test_docs_index.py` | `docs/index/` names every package, script and tool, and no path that has gone |

```bash
docker compose run --rm -T test python -m pytest -q                      # BVH half, ~2 min
docker compose run --rm -T test python -m pytest -q tests/test_roundtrip.py \
    --dataset truebones -k "Camel/bite"                                  # one clip
docker compose run --rm -T fbxsdk python -m pytest -q -k Camel           # BVH + FBX, one rig
```

The FBX half needs the Autodesk FBX SDK, which only the `fbxsdk` container has
(see [tooling](../../index/tooling.md#dependencies)); elsewhere it skips, and says
so. Under QEMU it costs about 11 s a clip, a few hours for the whole corpus.

`--dataset <name>` reads `data/<name>/` (prepared), `data/_raw/<name>/source/`
(raw) and `configs/dataset/<name>.yaml` (feature schema). Each test is one
clip, with id `<Rig>/<action>`.

## The round-trip test

Each clip goes prepared `.npz` → reduced model view → features → `fk` → back,
then [`RigRecord.restore`][poseydon.export.rig.RigRecord.restore] and [`BVH.from_animation`][poseydon.io.bvh.BVH.from_animation] → a file on disk → read back. The file
is compared against the **raw source BVH**, the one reference the pipeline did
not produce. It asserts:

- names, hierarchy, `CHANNELS` layout, frame count, frame rate and the OFFSET
  block all match the source;
- the root's world trajectory matches (the promoted root, on promoted rigs);
- every observable joint's world rotation matches, to 1e-3 degrees;
- non-root joints don't translate;
- in both rotation layouts (`rot6d_joint` and `rot6d_bone`), features rebuild
  the model view's positions and every rotation that orients something; the
  `rot6d_bone` reduction makes no real joint a leaf, `rot6d_joint` drops no
  joint but an End Site, and nothing is folded into a joint but the root.

`test_up_axis` turns one real rig to Z-up, declares `up_axis: +Z`, and checks
that preparation gives the same clip as the Y-up original and that inverting
returns the Z-up source.

The FBX test proves a chain of equalities. The rig's vendor FBX and raw BVH of
one clip calibrate to within 1e-4 of the extent, checked on every frame. The
restored clip written as FBX, read back, holds exactly the restored clip on the
vendor's node hierarchy, to 1e-4 (FBX keys are float32, and their rounding adds up along a long chain: 1.35e-5 at the tip of Anaconda's 27-joint spine). The prepared clip and the model view, written
onto the same asset in canonical space, read back with every joint where the
prepared clip (for the model view, its expansion) has it. With the BVH test showing that the
restored clip is the source, that puts the source back into the user's FBX.

Neither test asserts the losses the round trip makes by design: per-joint
translation, the locator chain above a promoted root (and Tukan's dead
subtree), and the root's own rotation where the reduction folded a joint into
it. A leaf's rotation is unobservable, so it isn't compared either.

On the 2026-10-13 corpus all 1145 Truebones clips pass the BVH test.

## Not covered

Models, sampling, training, losses, augmentation, conditioners, ops,
rendering, and feature parity with the reference. Add a test when one of them
is what you're troubleshooting.

## Passing is not the same as covering

A test that can't fail proves nothing, so break the behaviour it covers and
watch it go red. On 2026-10-12, the round-trip test failed under each of these
mutations: `FaceAxis.invert` as a no-op, the collapse fold removed from
`_remove`, every joint written in the default Euler order, and [`PromoteRoot`][poseydon.build.prepare.PromoteRoot]
reinserting the old root without the chain offset.

Two mutations it can't see, because this corpus never exercises the code:
writing BVH MOTION columns in joint-index order rather than depth-first order
(every rig read from a BVH is already depth-first), and dropping the FBX
pre/post-rotation correction (no Truebones node has one). The second was
removed instead: `poseydon.io.fbx` refuses a node with pre/post rotation.

An earlier suite learned the same lesson. The whole diffusion core survived
eleven separate mutations untouched, including signal and noise swapped in the
forward process.
