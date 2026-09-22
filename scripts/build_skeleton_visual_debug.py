"""Collect one comparable bundle per rig: source, prepared, model and roundtrip.

The bundle's whole value is that everything in it describes the SAME clip, so
`original.bvh` and `roundtrip.bvh` can be opened side by side. The model-space
animation is therefore DERIVED from the chosen action here rather than carried
over from whatever a previous run left in the directory: the sample is chosen
by what the raw tree happens to pair, that choice moves as the tree changes,
and an inherited `training.anim.npz` silently belonged to a different action in
60 of 73 bundles -- a roundtrip of one clip filed next to the source of
another, which reads as a badly broken export rather than a mismatched pair.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from poseydon.build.index import action_slug, is_all_takes_bundle, strip_skeleton_prefix


def _sha(path):
    if path is None or not Path(path).is_file(): return None
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()

def _sample(rig, corpus, raw):
    # The canonical raw tree is the only source of vendor representations.
    # Legacy ``raw/bvh`` and ``raw/fbx`` mirrors are intentionally ignored.
    source = raw / "source" / rig
    bvh=sorted(p for p in source.glob("*.bvh") if not is_all_takes_bundle(p.stem,rig))
    fbx=sorted(p for p in source.glob("*.fbx") if not is_all_takes_bundle(p.stem,rig))
    if not bvh: return None
    by_bvh={strip_skeleton_prefix(action_slug(p.stem),rig):p for p in bvh}
    by_fbx={strip_skeleton_prefix(action_slug(p.stem),rig):p for p in fbx}
    common=sorted(set(by_bvh)&set(by_fbx))
    if common: return common[0], by_bvh[common[0]], by_fbx[common[0]]
    return strip_skeleton_prefix(action_slug(bvh[0].stem),rig), bvh[0], (fbx[0] if fbx else None)

def build_bundle(contexts, output):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    for context in contexts:
        rig=context["rig"]; dest=output/rig; dest.mkdir(parents=True,exist_ok=True)
        metadata={"rig":rig,"action":context.get("action"),"unavailable":{}}
        for field in ("original_bvh","original_fbx","prepared_bvh","prepared_fbx","presentation_fbx","roundtrip_bvh","roundtrip_fbx","training_anim","training_features"):
            source=context.get(field)
            names={"training_anim":"training.anim.npz","training_features":"training.features.npz","roundtrip_bvh":"roundtrip.bvh","roundtrip_fbx":"roundtrip.fbx","original_bvh":"original.bvh","original_fbx":"original.fbx","prepared_bvh":"prepared.bvh","prepared_fbx":"prepared.fbx","presentation_fbx":"char-full.fbx"}
            target=dest/names[field]
            if source and Path(source).is_file():
                if Path(source).resolve() != target.resolve(): shutil.copy2(source,target)
                metadata[field]={"path":str(target),"sha256":_sha(target)}
            else:
                metadata[field]=None
                if field.endswith("fbx"): metadata["unavailable"][field]="source_surface_missing"
        (dest/"metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")

def _training(corpus: Path, rig: str, prepared: Path, folder: Path):
    """The model-space animation and features for THIS clip.

    Derived, not inherited: the same reduction `CharacterExporter` applies,
    against the same prepared clip the rest of the bundle describes. Returns
    ``(animation_path, features_path)``; the features are optional, because a
    manifest whose facing or foot joints did not survive reduction cannot
    resolve one, and an absent file is the honest record of that.
    """
    import numpy as np

    from poseydon.augment.topology import _reindex_resolved
    from poseydon.core.animation import RigidBodyAnimation
    from poseydon.core.rotations import QUAT_IDENTITY
    from poseydon.core.skeleton import SkeletonManifest, resolve
    from poseydon.features import DEFAULT_FEATURES, extract_features
    from poseydon.features.reduce import apply_reduction, build_reduction
    from poseydon.io.bvh import BVH

    with np.load(corpus / "rigs" / rig / "skeleton.npz", allow_pickle=True) as data:
        names = tuple(str(x) for x in data["names"])
        rest = RigidBodyAnimation.from_root_motion(
            rotations=np.tile(QUAT_IDENTITY, (1, len(names), 1)),
            root_pos=np.zeros((1, 3)), offsets=data["offsets"],
            parents=data["parents"].astype(np.int32), names=names, fps=1.0)
    reduction = build_reduction(rest)
    animation = BVH.read(prepared).to_animation().as_rigid_body(joint_translation="drop")
    reduced = apply_reduction(animation, reduction)
    folder.mkdir(parents=True, exist_ok=True)
    animation_path = folder / "training.anim.npz"
    reduced.save(animation_path)

    features_path = folder / "training.features.npz"
    try:
        manifest = SkeletonManifest.load(corpus / "rigs" / rig / "manifest.yaml")
        old_to_new = {old: new for new, old in enumerate(reduction.source_of)}
        resolved = _reindex_resolved(resolve(manifest, animation.names), old_to_new)
        features, spec = extract_features(reduced, resolved, DEFAULT_FEATURES)
        np.savez(features_path, features=features, names=np.array(spec.names, dtype=object))
    except Exception as error:  # noqa: BLE001 - features are optional here
        print(f"{rig}: no training features ({type(error).__name__}: {error})")
        features_path.unlink(missing_ok=True)
        features_path = None
    return animation_path, features_path


def discover(corpus, raw, output):
    contexts=[]
    for rigdir in sorted((Path(corpus)/"rigs").iterdir()):
        if not rigdir.is_dir(): continue
        sample=_sample(rigdir.name,Path(corpus),Path(raw))
        if sample is None: continue
        action,bvh,fbx=sample
        prepared=Path(corpus)/"clips"/rigdir.name/(action+".bvh")
        context={"rig":rigdir.name,"action":action,"original_bvh":bvh,"original_fbx":fbx,"prepared_bvh":prepared if prepared.is_file() else None}
        folder=Path(output)/rigdir.name
        if prepared.is_file():
            try:
                animation,features=_training(Path(corpus),rigdir.name,prepared,folder)
                context["training_anim"]=animation
                if features is not None: context["training_features"]=features
            except Exception as error:  # noqa: BLE001 - one rig must not cost the corpus
                print(f"{rigdir.name}: no training animation ({type(error).__name__}: {error})")
        # The roundtrips are the Blender companion's output, derived from the
        # training animation just rewritten -- so anything already here
        # describes a clip this run may have replaced. Dropped rather than left
        # to look current; the companion writes them next.
        for filename in ("roundtrip.bvh","roundtrip.fbx"):
            (folder/filename).unlink(missing_ok=True)
        for field,filename in (("prepared_fbx","char-model.fbx"),("presentation_fbx","char-full.fbx")):
            candidate=Path(corpus)/"rigs"/rigdir.name/filename
            if candidate.is_file(): context[field]=candidate
        contexts.append(context)
    return contexts

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--corpus",type=Path,default=Path("data/truebones"))
    parser.add_argument("--raw",type=Path,default=Path("data/_raw/truebones"))
    parser.add_argument("--out",type=Path,default=Path("tmp/skel_visual_debug"))
    args=parser.parse_args(argv)
    contexts=discover(args.corpus,args.raw,args.out); build_bundle(contexts,args.out)
    print(f"wrote {len(contexts)} rig bundles to {args.out}")
if __name__=="__main__": main()
