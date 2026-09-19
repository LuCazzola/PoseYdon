import json
from pathlib import Path
from poseydon.io.fbx import FBX
root=Path("data/truebones/rigs")
for kind in ("model_character.fbx","presentation.fbx"):
 paths=list(root.glob(f"*/{kind}")); meshes=0
 for p in paths:
  s=FBX.read(p); meshes += bool(s.meshes)
 print(kind,len(paths),meshes)
reports=list(root.glob("*/preparation-report.json")); complete=0
for p in reports:
 d=json.loads(p.read_text()); complete += all(k in d for k in ("authority","source_skeleton","model_skeleton","presentation_skeleton","outputs","availability","operations"))
print("reports",len(reports),"complete",complete)
