# Map

Where each piece of code lives. Grouped by what the code is for, not by folder. Each
library package has an API page under **API** in the nav.

## Library (`src/poseydon/`)

| Area | Path | What it holds |
|---|---|---|
| Contracts | `src/poseydon/core` | `Animation` / `RigidBodyAnimation`, skeleton manifests, `FeatureSpec`, the `MotionBatch`, rotations, FK (numpy and torch), topology relations, the `Registry` |
| File formats | `src/poseydon/io` | BVH (`bvh.py`), FBX through the Autodesk SDK (`fbx.py`), stick-figure MP4 (`render.py`), camera and floor framing (`framing.py`) |
| Corpus build | `src/poseydon/build` | stage 1 sources, clip filters, invertible prepare stages, rig cleanup, joint-name labels; stage 2 passes and the clip index |
| Legacy ingest | `src/poseydon/ingest` | `poseydon ingest` (one-step align and index), plus `align.py` helpers that `build`, `features` and `training` import |
| Features | `src/poseydon/features` | feature blocks, extraction, reduction (the model's lens), recovery and reconstruction back to a skeleton |
| Data loading | `src/poseydon/data` | dataset, windows, per-skeleton normalization, rig-balanced sampler, collate |
| Augmentation | `src/poseydon/augment` | structural augmentations (drop or duplicate joints), an `apply_features` hook no shipped augmentation uses yet, and `JointEdit` transport |
| Conditioning | `src/poseydon/conditioners` | topology, T-pose, normalization stats, joint names; T5 name embeddings |
| Models | `src/poseydon/models` | `Denoiser` / `LatentDenoiser`, AnyTop, MoDiffAE; `modules/` holds graph attention, backbone, masking, embeddings |
| Processes | `src/poseydon/process` | Gaussian diffusion and flow matching |
| Sampling | `src/poseydon/sampling` | DDPM, DDIM and Euler samplers, classifier-free guidance; `controls/` holds inbetween, latent pin and latent mix |
| Operations | `src/poseydon/ops` | generate, inbetween, blend and retarget; `retarget_run.py` is the shared end-to-end retarget |
| IK | `src/poseydon/solvers` | gradient IK and its objective terms |
| Losses | `src/poseydon/losses` | simple, geodesic, foot skate, KL |
| Training | `src/poseydon/training` | config to objects (`build.py`), `MotionTask`, Lightning wrappers, retarget validation and its metrics, the checkpoint recipe |
| Export | `src/poseydon/export` | `RigRecord`: moving a clip between model view, prepared and source |
| CLI | `src/poseydon/cli.py` | `poseydon ingest / train / sample / retarget / export / list` |

## Tooling

| Area | Path | What it holds |
|---|---|---|
| Corpus pipeline | `scripts/process_dataset_truebones.py`, `scripts/build_features.py` | stage 1 (raw to prepared) and stage 2 (features, stats, index) |
| FBX inputs | `scripts/build_fbx_cache.py`, `tools/build_rest_fbx.py`, `scripts/fetch_fbx_sdk.sh` | read vendor FBX once into a cache, build each rig's rest asset, fetch the SDK |
| Dataset authoring | `tools/write_rest_pose.py`, `tools/tpose_author.py`, `tools/write_unimate_manifests.py`, `scripts/fetch_unimate_annotations.sh`, `tools/build_joint_name_cache.py` | rest-pose choice, authored T-poses, UniMate manifests and inputs, the T5 joint-name cache |
| Rendering | `scripts/render_fbx.py`, `scripts/build_stage_samples.py`, `tools/render.py` | skinned MP4 via Blender; one clip per rig at every round-trip stage; a re-export of the stick-figure renderer for the parity tools |
| Motion verifier | `scripts/verify_motion.py`, `scripts/verify_motion.sh`, `tools/verify`, `configs/verify` | judge returned FBX against the vendor rig; `tools/verify/sheet.py` draws the mesh sheets (also used by `tools/tpose_author.py`) |
| Reference parity | `tools/reference`, `external/neural_motion_blending` | compare against the published implementation (read-only checkout), `compat` image only |
| Probes | `tools/probe_*.py`, `tools/ik_*.py` | one-off investigations (rig shapes, rest poses, IK settings, parity runs against the reference weights) |
| Configs | `configs` | Hydra groups `data`, `features`, `model`, `process`, `trainer`, plus `train.yaml`, `sample.yaml`, `build.yaml`; `verify/` is plain YAML read by the verifier |
| Data | `data` | per dataset: `rigs/<Rig>/manifest.yaml` and `clips/<Rig>/<action>.yaml` labels (tracked for `truebones` only; `truebones_bone` and `unimate` are ignored whole); prepared clips, caches and raw vendor files (ignored) |
| Tests | `tests` | `test_roundtrip.py`, `test_verify.py` (the docs are checked by `docs/parallax/tools/check_docs.py`) |
| Containers | `Dockerfile`, `docker-compose.yml` | services `test`, `train`, `compat`, `fbx` (Blender), `fbxsdk`, `docs` |
| Docs | `docs/parallax` | this site; `tools/` holds the docs check, preview and ticket scripts |
