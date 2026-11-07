# Issues & Ideas

A ticket is a bug, loose end or idea that was noticed but not acted on.

- **Issues** (bugs and loose ends) live in [GitHub Issues](https://github.com/LuCazzola/PoseYdon/issues),
  labelled by priority tier and `needs-triage` until reproduced
  ([untriaged](https://github.com/LuCazzola/PoseYdon/issues?q=is%3Aopen+label%3Aneeds-triage)):
    - `Prio:CRITICAL`: breaks a run, or may harm data quality or integrity.
    - `Prio:WANTED`: relevant fixes and features worth doing, but not urgent.
    - `Prio:MINOR`: small, cheap fixes: stale references, noisy logs, cosmetic gaps.
- **Ideas** (proposed changes, pending decisions and things worth investigating;
  nothing is broken yet) live in the [table below](#ideas), in the repo, so they
  are reviewed in the same PR as the code they concern.

Open, edit and close tickets (ideas and issues alike) only with these
scripts, run on the host where `gh` is logged in:

```bash
python3 docs/reference/open_ticket.py <critical|wanted|minor|idea> "<issue>" <path> [<path> ...] \
    [--title "<title>"] [--repro "<command>"] [--cause "<guess>"]
python3 docs/reference/edit_ticket.py <ID|#N> [--tier <tier>] [--issue "<issue>"] [--where <path> ...] \
    [--repro "<command>"] [--cause "<guess>"] [--verified "<evidence>"]
python3 docs/reference/close_ticket.py <ID|#N> [...] [--reason "not planned"] [--comment "<why>"]  # --comment required for issues
```

- An idea gets a 4-character ID (`grep <ID> docs/reference/ideas.md`), an
  issue a number (`gh issue view <N>`).
- An issue keeps apart what was **Observed** (`<issue>`: facts only), how to
  **Reproduce** it (`--repro`: a command with its dataset, rig and overrides) and
  the **Suspected cause** (`--cause`), because the guessed cause is the part most
  often wrong. Give `--repro` whenever a command shows the bug; an empty value
  removes a section.
- Every new issue carries `needs-triage`. Drop it with `--verified` only on
  evidence: a red macro test, or a command and its output.
- A fix closes its issue through `Fixes #N: <what fixed it>` in the commit
  message or PR, one line per issue, so the issue's "closed in" link explains
  itself. `close_ticket.py` is for issues that turn out wrong or not planned,
  and needs `--comment` saying why.
- Moving a ticket between an idea and an issue (`--tier`) opens it in the new
  place and closes it in the old one.

!!! warning "Leads, not facts"
    Each idea and issue is what was believed when it was written. It may be
    stale, already fixed, wrong about the cause or narrower than it looks.
    Before acting on one, check it against the code and the data, and fix
    the ticket if it disagrees.

## Ideas

Newest first.

| ID | Noticed | Where | Issue |
|---|---|---|---|
| `973y` | 2026-11-07 | `tools/build_rest_fbx.py`, `src/poseydon/io/` | Rest assets keep the vendor's bind, not the T-pose: since 2026-11-07 `rest.fbx` is the vendor FBX with one frame at the rest pose (`FBX.pose`), so the mesh is exactly the vendor's skinned into any pose, but the file's bind pose stays the vendor's: a mesh's coordinates are bind-relative, not rest-relative, and a tool that shows a file's bind pose shows the vendor's. Chosen for now; the owner decides later. Making the rest pose the bind (`FBX.rebind`, removed here, last at commit 89bff67) stores the mesh in the T-pose but moves blended vertices, linear blend skinning not composing: up to 5.4e-2 of the extent (Goat), above 1e-3 on 23 of 73 rigs |
| `mclf` | 2026-10-21 | `src/poseydon/build/sources.py` | `fbx_cache` does not model Blender's bone re-orientation: rest rotations are the FBX bind's; whether this matches UniMate's published rest poses is unmeasured. Unit scale no longer differs in effect: since 2026-10-21 a node's world scale is taken off before its local transform is read, so offsets and the root's travel are both in world units, where Blender keeps both in armature units; the two differ by one uniform factor, which the `scale` stage normalizes away |
| `d7pn` | 2026-10-21 | `configs/data/truebones.yaml` | 49 rigs train on their T-pose file as a motion clip: the index holds 48 `tpose` clips and Parrot2's `parrot_tpose` with `split: train`, so balanced sampling draws static T-pose windows as motion (2 to 92 frames of one pose). Read from FBX since 2026-10-21, seven vendor T-poses are one frame and left the index (Bat, Bear, Crow, Elephant, FireAnt, Fox, Spider: their BVHs held 31 to 91 static frames). Whether to keep the rest is pending |
| `rs5y` | 2026-10-21 | `data/truebones/fbx_clips.csv` | FBX-only Truebones clips are not in the corpus: `data/truebones/fbx_clips.csv` keeps the BVH-era clip set; vendor FBX with no BVH (KingCobra `run`, `walk`, `walk fast`, and the takes inside each rig's all-takes bundle, Fox `idle_1` and Lynx `idle4` among them) are left out. Whether to add them is pending |
| `qwmj` | 2026-10-21 | `tools/tpose_author.py`, `tools/build_rest_fbx.py` | Truebones T-poses are still authored in BVH: `tools/tpose_author.py` writes `__TPOSE_authored.bvh`, and `tools/build_rest_fbx.py` carries a manifest's `rest_pose:` BVH onto the FBX nodes through a vendor BVH/FBX pair of one clip: the one place the vendor BVHs are still read. Authoring on the FBX skeleton directly would drop them; not done |
| `wb8q` | 2026-10-20 | `src/poseydon/build/cleanup.py` | `prune_unskinned` restores a removed joint at bind, not at its clip pose: stillness is measured from frame 0 but `invert` restores at bind; a removed joint that held a constant angle off bind moves its removed descendants by an unbounded constant (Scorpion2 `jt_Tail09x_C`, 2.61 degrees; its child is in `PRUNED_SKIP`). Pending decision: restore at the clip's frame-0 pose or keep bind |
| `awc0` | 2026-10-20 | `src/poseydon/build/sources.py` | One bind pose per rig is assumed, not measured per clip: `FbxCache.rest` takes the first main-group clip's bind; the per-clip bind-pose difference (spec: one bind pose per rig) has not been measured, and the 71 dropped training clips (low_activity 68, jump 3) are all explained by the filters and add up to the published 1023 / 95491 |
| `qo8h` | 2026-10-20 | `src/poseydon/build/sources.py` | `data=unimate` rest frame is the bind pose, not a neutral pose: stage 1 writes the prepared bind pose as `clips/<Rig>/bind_pose.bvh` and `rest_action` returns `bind_pose`, so the `rest_frame` conditioner is the bind pose's frame 0 (T-pose or not, per rig). Foot joints (no `foot_joints` in the generated manifests) and labels are left at PoseYdon defaults: modelling choices for training |
| `aa2e` | 2026-10-20 | `scripts/process_dataset_truebones.py`, `src/poseydon/build/pipeline.py` | Stage 1 hands stage 2 a BVH, which loses data the `.npz` could keep: `scripts/process_dataset_truebones.py` writes `clips/<Rig>/<action>.bvh` and stage 2's `build_clips` (`src/poseydon/build/pipeline.py`) reads it back into the `.npz` training reads. The text round trip is what makes BVH re-list joints depth-first (`bfs_order`) and `index.jsonl` read fps 30.000000030000002 (the 10-decimal frame time), and needs `leaf_rotations` to keep a childless joint's rotation; Euler angles in text also round. Writing the `.npz` (facing included) in stage 1 would drop that hand-off. Other readers of the stage-1 BVH: stage 2's skeleton pass (the rest clip), stage 1's own sanity check, `scripts/build_stage_samples.py` (`2_prepared.bvh`) and `tools/probe_resolve_after_reduce.py`; a BVH for viewing could be written on demand. Rebuilds the corpus. truebones clips: 181 MB BVH beside 141 MB `.npz` |
| `no52` | 2026-10-19 | `tools/tpose_author.py` | 15 rigs' rest pose is now an authored T-pose: until 2026-10-19 `rest_pose:` named a clip's first frame for Anaconda (coiled), Ant, Bird, Camel, Crab, Cricket, Goat, Jaguar, Lion, Monkey, Pteranodon, Rat (`__Trottle`, mid-trot; no T-pose shipped), SabreToothTiger (`__Startwalk`, seated), Scorpion (its `__TPOSE.bvh` is asymmetric) and Trex; Deer, Dog and Scorpion-2 take a clip frame that is already symmetric. That frame is what `RestRelative` makes identity and what the `tpose` conditioner shows the model. Ant, Crab, Deer and Jaguar ship a `__TPOSE.bvh` that cannot replace it: each lacks 5-10 leaf joints the clips have, so `rest_source` rejects it, and its offsets differ from the clips'. Rest-pose asymmetry (mirrored mismatch, median over pairs, fraction of extent): Anaconda 0.47, Rat 0.39, Monkey 0.20, Crab 0.19, Bird 0.13, Trex 0.09, others 0.03-0.08. Reviewed authored T-poses now sit in the raw folder as `__TPOSE_authored.bvh` for all 15 (`tools/tpose_author.py`): Anaconda, Ant, Bird, Camel, Crab, Cricket, Goat, Jaguar, Lion, Monkey, Pteranodon, Rat, SabreToothTiger, Scorpion and Trex (one frame, the clips' joint set and channels); their manifests now point `rest_pose:` (and `artifacts.bvh`) at them, and `rest_source` accepts each; the corpus was rebuilt on them on 2026-10-19, so checkpoints trained earlier read different rest poses for these 15. Each is one frame, so it is written as a clip but left out of the index (`MIN_CLIP_FRAMES` in `src/poseydon/build/pipeline.py`: velocity needs two frames), and does not train as motion. Goat's upper-arm helper bones (`BN__UpperArm_L_01`/`_R_01`) cannot be mirrored: the vendor gave them different offsets (8.6 vs 6.6 long). Lion's hip joints sit 3.5 off-mirror for the same reason (thigh offsets 13.6 vs 11.6): the right hip joint lies 3.7 inside its own thigh mesh (the thigh meshes mirror each other within 0.3), and its chest mesh sits 1.9 to the right of the neck bone, so its pelvis and belly stay slightly lopsided. SabreToothTiger's two front paw meshes differ (in no clip frame do all of the left paw's toes touch the floor), so its front paws and fingers are posed per side and are not exact mirrors; the rest of its skeleton is. |
| `kux0` | 2026-10-16 | `data/truebones/rigs/Scorpion/manifest.yaml` | Scorpion's motion is linearly keyed: 21% of its moving joint-frames have near-zero acceleration (constant speed between keys), against a corpus median of 1%; next are FireAnt and Scorpion-2 at 7%. Rat and Roach, which look stiff, are not (rationale for their stiffness not found; Rat's dropped translations are the `EnforceRigid` issue) |
| `u7dr` | 2026-10-16 | `configs/data/truebones.yaml` | Single-frame jumps in some clips: Monkey `die` starts on an exact copy of the rest pose (0.0° from it), then jumps 6.6 bone lengths; Hound `die`, Scorpion-2 `death_2` and Spider `scary` jump 3-6 bone lengths between frame 0/1 and the next; mid-clip: Deer `grazing` frame 269 (4.3), Monkey `idle3` frame 139 (3.3), SpiderG `attack4`/`attack6`, KingCobra `attack2`, Flamingo `onelegbent` frame 12 (10.1, a validation pair's content clip). Parrot2 `parrot_land` moves 29x faster than its rig's median clip. Not inspected visually |
| `pim1` | 2026-10-16 | `src/poseydon/build/clips.py` | Static and held frames in the training clips: Dog `idle`, Fox `idle_1` and Scorpion-2 `idle` do not move at all (91 frames each); 31 clips end (or start) on 10 or more frames of one held pose, 1415 of 110611 frames: Pteranodon's flight clips hold the last 45 of 91 frames (a third of the rig's frames), Raptor3 `run` the last 57, SabreToothTiger `180right` the last 111 of 181, plus Pigeon and Camel `fall`. Some are a death pose held on purpose; candidate: trim held ends to a few frames in stage 1 |
| `lmd8` | 2026-10-14 | `data/truebones/rigs/Horse/manifest.yaml` | Horse `knockedback` is far outside its rig's statistics: 30 frames; `local_vel` reaches 30σ and `ric_pos` 9σ on both corpora, clip mean square about 8 against a corpus median under 1; near step 134k of run h2pkinn4 its gradient norm at t=5 is 50 (typical 6). Not inspected visually |
| `uf6c` | 2026-10-14 | `src/poseydon/features/kinematics.py` | Turning clips put `heading` 8–11σ from its rig's mean: `heading` is normalized per (joint, channel) like every block, and most clips barely turn from their first frame, so `sin(yaw)` has a small std per rig; Trex `turn_180_*`, Lion `die2`, Monkey `die` reach 8–11σ on the bone-relative corpus. Whether to scale `heading` differently (e.g. `scale: none`, since it is already in [-1, 1]) is pending |
| `6xr8` | 2026-10-14 | `src/poseydon/features/kinematics.py`, `src/poseydon/losses/base.py` | Root-only channels are zero padding on every other joint, and the padding is noised and trained on: `heading` (`src/poseydon/features/kinematics.py`) is `(cos, sin)` on the root and zero elsewhere, but diffusion noises every channel and `element_mask` (`src/poseydon/losses/base.py`) masks joints and frames, not channels; so `joint_embedding` reads noise in those columns and `joint_dembedding` is trained to output zero there. The models already embed and project the root separately (`src/poseydon/models/modiffae.py`, `src/poseydon/models/anytop.py`). Measured on run h2pkinn4 after 45k steps: the `heading` block is 0.006 of `simple` 1.75, so the padding slightly lowers `simple` against runs without the block. Candidate fix, pending the owner's check: a per-(joint, channel) mask from the spec, applied to the noise and the loss; `foot_contact` on non-foot joints is a similar case, though there the zero is a real "not a foot" value |
| `j228` | 2026-10-13 | `src/poseydon/solvers/gradient_ik.py` | `GradientIK` starts every joint at identity (`src/poseydon/solvers/gradient_ik.py`): on the bone-relative ablation corpus identity is a straight chain, not the rest pose, so IK starts from a worse guess there; not measured |
| `gart` | 2026-10-13 | `data/truebones/rigs/Dog/manifest.yaml` | Dog's facing comes from thighs + upper arms (`data/truebones/rigs/Dog/manifest.yaml`): measured against pelvis-to-head on frame 0 of 41 clips: median 1.8°, max 21°, 4 clips over 10°. Clavicles + hands measured worse (hands swing with the gait: 14 clips over 10°); clavicles alone measured best (median 0.2°, max 16.5°). Awaiting the owner's visual inspection; a change needs `scripts/process_dataset_truebones.py --rigs Dog` |
