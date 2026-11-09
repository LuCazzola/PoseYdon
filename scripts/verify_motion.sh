#!/usr/bin/env bash
# The verifier in one command: build in `fbxsdk`, read in Blender (`fbx`), judge in `fbxsdk`.
#   scripts/verify_motion.sh [--data D] (--rig R [--clip C] | --smoke) [--out DIR]
# Runs on the host, only calls docker compose. A rig that fails to build is reported and the others
# still run; the exit status is non-zero if any build or judge was.
set -uo pipefail
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
code=0
built=$("${sdk[@]}" build "${args[@]}" --out "$out") || code=1
grep FAILED <<<"$built" >&2
for run in $(grep -v FAILED <<<"$built"); do
  [ -f "$run/case.json" ] || continue  # build prints a run directory per rig, and `<rig>: FAILED ...` lines
  docker compose run --rm -T fbx blender -b --python tools/verify/blender_read.py -- "$run" >/dev/null || code=1
  "${sdk[@]}" judge "$run" || code=1
done
exit $code
