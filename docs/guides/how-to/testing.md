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
(raw) and `configs/build.yaml` composed with `data=<name>` (the dataset's
chain and labels, and the feature blocks). Each test is one
clip, with id `<Rig>/<action>`.

## The round-trip test

Each clip goes prepared `.npz` → reduced model view → features → `fk` → back,
then [`RigRecord.restore`][poseydon.export.rig.RigRecord.restore] and [`BVH.from_animation`][poseydon.io.bvh.BVH.from_animation] → a file on disk → read back. The file
is compared against the **raw source**, the one reference the pipeline did
not produce. The test reads it through the dataset's own `source:` (the FBX
cache for `data=truebones` and `data=unimate`, or BVH files) and cuts it to the
frames the clip kept (`clip_params[action]["frames"]["window"]`, absent meaning
the whole clip). It asserts:

- names, hierarchy, frame count and frame rate match the source, and for a BVH
  source the `CHANNELS` layout and the OFFSET block too;
- for an FBX source, the reader itself: every joint whose FBX local translation
  never leaves its rest offset sits where the SDK's cached world matrices put it
  (this caught the Crow scale bug: with an armature scaled 5, the reader put its
  joints 15 extents off);
- the root's world trajectory matches (the promoted root, on promoted rigs);
- every observable joint's world rotation matches, to 1e-3 degrees;
- non-root joints don't translate;
- a joint `prune_unskinned` removed (`data=unimate`) sits, against its nearest
  kept ancestor, where the source made rigid the way `EnforceRigid` does puts
  it, to the largest distance a joint turning less than 2 degrees can travel
  (`2 d sin(1 deg)` per removed ancestor at distance `d`); its world rotation
  is not compared. `PRUNED_SKIP` names the one exception (Scorpion2
  `jt_Tail10_C`: its removed parent sits 2.6 degrees off its bind rotation in
  every clip, because pruning measures the turn from frame 0);
- in both rotation layouts (`rot6d_joint` and `rot6d_bone`), features rebuild
  the model view's positions and every rotation that orients something; the
  `rot6d_bone` reduction makes no real joint but a still one a leaf,
  `rot6d_joint` drops no joint but an End Site or a still joint, and nothing
  is folded into a joint but the root; still joints come back within the
  `drop_still` rule's `degrees` (as `skeleton.npz` records it), every other
  joint exactly.

`test_up_axis` turns one real rig to Z-up, declares `up_axis: +Z`, and checks
that preparation gives the same clip as the Y-up original and that inverting
returns the Z-up source. It reads the rig through the dataset's own source and
gives a stage that decides from the whole rig (`prune_unskinned`) the rig's
clips and skin, as stage 1 does.

`test_round_trip_fbx` skips a dataset whose source is not FBX (there is no source
FBX to write back onto). It writes the restored clip onto the rig's rest asset
(`RigRecord.asset`) by node name, reads the file back and compares it with the
source FBX itself (the cached world matrices of the clip): the root's position,
and the orientation of every joint the round trip keeps, leaves included (an FBX
node holds its own rotation, and the mesh shows it; this caught stage 1 losing
every leaf's rotation on the FBX-read corpus), to
1e-4 (FBX keys are float32, and their rounding adds up along a long chain); node
scale within 1% (it is the asset's: Anaconda's files disagree on it by 0.4% from
clip to clip). The prepared clip and the model view, written onto the same asset
in canonical space (`RigRecord.placement`), read back with every joint where the
prepared clip has it. Joints held by structure (`RigRecord.fixed`) are written
undriven. The mesh is not compared: the rest asset's is the vendor mesh re-bound
to the rest pose ([export](../../api/export/index.md#limitations)).

Neither test asserts the losses the round trip makes by design: per-joint
translation, the locator chain above a promoted root (and its dead branches:
Tukan's, Crow's), and the root's own rotation where the reduction folded a
joint into it. A leaf's rotation is unobservable, so it isn't compared either.

On the 2026-10-21 corpus, read from FBX, all 1135 Truebones clips pass the BVH
test, as do all 1135 of `truebones_bone` and all 1023 `data=unimate` training
clips (`--dataset unimate`).

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
