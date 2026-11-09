#!/usr/bin/env bash
# The verifier in one command: build in `fbxsdk`, read in Blender (`fbx`), judge in `fbxsdk`.
#   scripts/verify_motion.sh [--data D] (--rig R [--clip C] | --smoke) [--out DIR]
# Exits with judge's status. Runs on the host; only calls docker compose.
set -euo pipefail
cd "$(dirname "$0")/.."
out=tmp/verify
args=()
while [ $# -gt 0 ]; do
  case $1 in
    --out) out=$2; shift 2 ;;
    *) args+=("$1"); shift ;;
  esac
done
sdk=(docker compose run --rm -T --entrypoint python -e HOME=/tmp fbxsdk scripts/verify_motion.py)
runs=$("${sdk[@]}" build "${args[@]}" --out "$out" | grep "^$out/")
for run in $runs; do
  docker compose run --rm -T fbx blender -b --python tools/verify/blender_read.py -- "$run" >/dev/null
done
code=0
for run in $runs; do
  "${sdk[@]}" judge "$run" || code=$?
done
exit $code
