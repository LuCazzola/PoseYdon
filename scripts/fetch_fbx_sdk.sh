#!/usr/bin/env bash
# Download the Autodesk FBX Python SDK into third_party/fbxsdk/ (gitignored).
#
# The SDK is proprietary: free to use, but it may not be committed or
# redistributed, so every user fetches their own copy and accepts Autodesk's
# license themselves. Read it first: https://aps.autodesk.com/developer/overview/fbx-sdk
#
#   scripts/fetch_fbx_sdk.sh --accept-license
set -euo pipefail

VERSION=2020311
ARCHIVE="fbx${VERSION}_fbxpythonsdk_linux.tar.gz"
URL="https://damassets.autodesk.net/content/dam/autodesk/www/files/${ARCHIVE}"
DEST="$(cd "$(dirname "$0")/.." && pwd)/third_party/fbxsdk"

if [[ "${1:-}" != "--accept-license" ]]; then
  echo "The FBX SDK is licensed by Autodesk: https://aps.autodesk.com/developer/overview/fbx-sdk" >&2
  echo "Re-run with --accept-license once you have read and accepted its terms." >&2
  exit 1
fi

mkdir -p "$DEST"
if [[ -f "$DEST/$ARCHIVE" ]]; then
  echo "already present: $DEST/$ARCHIVE"
else
  curl -fL --retry 3 -o "$DEST/$ARCHIVE.part" "$URL"
  mv "$DEST/$ARCHIVE.part" "$DEST/$ARCHIVE"
  echo "downloaded: $DEST/$ARCHIVE"
fi
