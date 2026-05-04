#!/usr/bin/env bash
#
# Regenerate tests/fixtures/cat_image.jpg from the upstream Wikimedia Commons source.
#
# Source: https://commons.wikimedia.org/wiki/File:2008-11-28_Calico_kitten_on_the_litter_box.jpg
# License: CC BY 2.0 — see LICENSES.md for full attribution (the bundled fixture is the
#          required attribution under CC BY 2.0; do not strip it from the repo).
#
# The bundled JPEG must be ≤30 KB so the repo doesn't bloat. The ffmpeg invocation below scales the
# original (2430x1620) down to 400 wide and re-encodes at quality 7, producing ~15 KB. Tune ``-vf
# scale=...`` and ``-q:v`` if the upstream image changes.
#
# Run from the repo root:
#   bash tests/fixtures/regenerate-cat-image.sh

set -euo pipefail

SOURCE_URL="https://upload.wikimedia.org/wikipedia/commons/c/cd/2008-11-28_Calico_kitten_on_the_litter_box.jpg"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TMP_ORIGINAL="$(mktemp -t cat-original.XXXXXX.jpg)"
OUTPUT="${SCRIPT_DIR}/cat_image.jpg"
MAX_BYTES=30720 # 30 KB

trap 'rm -f "${TMP_ORIGINAL}"' EXIT

echo "Downloading: ${SOURCE_URL}"
curl --fail --silent --show-error --location --output "${TMP_ORIGINAL}" "${SOURCE_URL}"

echo "Scaling + recompressing to ${OUTPUT}"
ffmpeg -y -hide_banner -loglevel error \
	-i "${TMP_ORIGINAL}" \
	-vf "scale=400:-1" \
	-q:v 7 \
	"${OUTPUT}"

actual_bytes="$(wc -c <"${OUTPUT}" | tr -d ' ')"
echo "Output size: ${actual_bytes} bytes (cap: ${MAX_BYTES})"
if [ "${actual_bytes}" -gt "${MAX_BYTES}" ]; then
	echo "ERROR: ${OUTPUT} exceeds ${MAX_BYTES} bytes; tune the ffmpeg flags above." >&2
	exit 1
fi

echo "Done."
