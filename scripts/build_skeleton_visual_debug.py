from __future__ import annotations
import argparse, hashlib, json, shutil
from pathlib import Path
from poseydon.build.index import action_slug, is_all_takes_bundle, strip_skeleton_prefix

def _sha(path):
    if path is None or not Path(path).is_file(): return None
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()

def _sample(rig, corpus, raw):
    bvh=sorted(p for p in (raw/"bvh"/rig).glob("*.bvh") if not is_all_takes_bundle(p.stem,rig))
    fbx=sorted(p for p in (raw/"fbx"/rig).glob("*.fbx") if not is_all_takes_bundle(p.stem,rig))
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
            names={"training_anim":"training.anim.npz","training_features":"training.features.npz","roundtrip_bvh":"roundtrip.bvh","roundtrip_fbx":"roundtrip.fbx","original_bvh":"original.bvh","original_fbx":"original.fbx","prepared_bvh":"prepared.bvh","prepared_fbx":"prepared.fbx","presentation_fbx":"presentation.fbx"}
            target=dest/names[field]
            if source and Path(source).is_file():
                if Path(source).resolve() != target.resolve(): shutil.copy2(source,target)
                metadata[field]={"path":str(target),"sha256":_sha(target)}
            else:
                metadata[field]=None
                if field.endswith("fbx"): metadata["unavailable"][field]="source_surface_missing"
        (dest/"metadata.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")

def discover(corpus, raw, output):
    contexts=[]
    for rigdir in sorted((Path(corpus)/"rigs").iterdir()):
        if not rigdir.is_dir(): continue
        sample=_sample(rigdir.name,Path(corpus),Path(raw))
        if sample is None: continue
        action,bvh,fbx=sample
        prepared=Path(corpus)/"clips"/rigdir.name/(action+".bvh")
        context={"rig":rigdir.name,"action":action,"original_bvh":bvh,"original_fbx":fbx,"prepared_bvh":prepared if prepared.is_file() else None}
        old=Path(output)/rigdir.name
        for field,filename in (("roundtrip_bvh","roundtrip.bvh"),("roundtrip_fbx","roundtrip.fbx"),("training_anim","training.anim.npz"),("training_features","training.features.npz")):
            candidate=old/filename
            if candidate.is_file(): context[field]=candidate
        for field,filename in (("prepared_fbx","model_character.fbx"),("presentation_fbx","presentation.fbx")):
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
