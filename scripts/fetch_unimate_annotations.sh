#!/usr/bin/env bash
# Fetch the UniMate inputs from the Hugging Face dataset Linzhan/Truebones-ZOO-Annotations
# into data/_raw/UniML3D/truebones_annotations/ (gitignored, under data/_raw/):
#
#   clips.csv               the Truebones clip index UniMate exports from
#   scripts/probes/*.py     the FBX repair scripts (strip variant, graft, texture embed)
#
# The commit is pinned so the files never change under the recipe. Also links
# data/_raw/unimate -> truebones, the path the UniMate dataset configs read.
#
#   scripts/fetch_unimate_annotations.sh
set -euo pipefail

REPO=Linzhan/Truebones-ZOO-Annotations
SHA=b127ca411ab15e9e3900a92d87f24778db1787a6
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/data/_raw/UniML3D/truebones_annotations"
BASE="https://huggingface.co/datasets/$REPO/resolve/$SHA"
FILES=(
  clips.csv
  scripts/probes/fbx_strip_variant.py
  scripts/probes/fbx_graft_mesh.py
  scripts/probes/fbx_embed_texture.py
)

for f in "${FILES[@]}"; do
  if [[ -f "$DEST/$f" ]]; then
    echo "already present: $DEST/$f"
    continue
  fi
  mkdir -p "$(dirname "$DEST/$f")"
  curl -fL --retry 3 -o "$DEST/$f.part" "$BASE/$f"
  mv "$DEST/$f.part" "$DEST/$f"
  echo "downloaded: $DEST/$f"
done

ln -sfn truebones "$ROOT/data/_raw/unimate"
echo "linked: $ROOT/data/_raw/unimate -> truebones"
