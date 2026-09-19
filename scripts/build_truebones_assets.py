"""Build all Truebones model/presentation FBX assets from canonical raw data."""
from pathlib import Path
import numpy as np
from poseydon.build.character_assets import CharacterAssetContext, CharacterAssets, build_character_assets, _write_report
from poseydon.build.index import is_all_takes_bundle
from poseydon.features.reduce import build_reduction, apply_reduction
from poseydon.core.animation import RigidBodyAnimation
from poseydon.core.rotations import QUAT_IDENTITY
from poseydon.io.fbx import FBX, _to_y_up, _rest_positions, _similarity
raw=Path("data/_raw/truebones/source"); out=Path("data/truebones/rigs")
Path("tmp/asset-fails.txt").write_text(""); count=0
for rigdir in sorted(out.iterdir()):
    if not rigdir.is_dir(): continue
    candidates=sorted(p for p in (raw/rigdir.name).glob("*.fbx") if not is_all_takes_bundle(p.stem,rigdir.name))
    if not candidates: continue
    with np.load(rigdir / "skeleton.npz", allow_pickle=True) as data:
        names=tuple(str(x) for x in data["names"])
        full=RigidBodyAnimation.from_root_motion(rotations=np.tile(QUAT_IDENTITY,(1,len(names),1)),root_pos=np.zeros((1,3)),offsets=data["offsets"],parents=data["parents"].astype(np.int32),names=names,fps=1.0)
    reduction=build_reduction(full); model=apply_reduction(full,reduction)
    target=_rest_positions(model); best=None
    for source in candidates:
        try:
            scene=FBX.read(source); scene.clear_animation(); indices=[scene.joint_names.index(n) for n in model.names]
            source_pos=np.stack([_to_y_up(scene.joint_position(n,rest=True)) for n in model.names])
            scale,rot,offset=_similarity(source_pos,target); fitted=scale*(source_pos@rot.T)+offset
            residual=float(np.max(np.linalg.norm(fitted-target,axis=1)))
            if best is None or residual<best[0]: best=(residual,source,tuple(scene.joint_names),tuple(scene.joint_parents))
        except Exception: continue
    if best is None:
        scene=FBX.read(candidates[0]); context=CharacterAssetContext(rig=rigdir.name,raw_fbx=candidates[0],model_names=tuple(scene.joint_names),model_parents=tuple(scene.joint_parents))
    else:
        residual,source,source_names,source_parents=best
        if all(n in source_names for n in model.names) and residual < 0.05:
            context=CharacterAssetContext(rig=rigdir.name,raw_fbx=source,model_names=tuple(model.names),model_parents=tuple(int(x) for x in model.parents),removals=reduction.ops,model_rest=model)
        else:
            # Explicitly fall back to the FBX-authoritative topology when the
            # paired bind geometry is incompatible; the report records this
            # source asset and the resulting model remains self-consistent.
            context=CharacterAssetContext(rig=rigdir.name,raw_fbx=source,model_names=source_names,model_parents=source_parents)
    try:
        assets=build_character_assets(context,rigdir); count += assets.model is not None
    except Exception as error:
        for stale in (rigdir / "char-model.fbx", rigdir / "char-full.fbx"):
            if stale.exists(): stale.unlink()
        context = locals().get("context")
        if context is not None: _write_report(context, rigdir, CharacterAssets(None, None, f"{type(error).__name__}: {error}"))
        Path("tmp/asset-fails.txt").open("a").write(f"{rigdir.name}: {type(error).__name__}: {error}\n")
print("built",count)
