#!/usr/bin/env bash
# Pre-commit hook companion for the ``forbid-binary-outside-fixtures`` entry. Receives a list of
# staged file paths matching the ``files`` regex and rejects any that live outside tests/fixtures/.
# Keeps the repo size bounded — binary blobs (mp4/dav/jpg/png/pt/onnx/etc.) belong on the external
# storage drive, not in git.

set -euo pipefail

exit_code=0
for path in "$@"; do
	case "$path" in
	tests/fixtures/*) ;;
	*)
		echo "forbidden binary outside tests/fixtures/: $path" >&2
		exit_code=1
		;;
	esac
done
exit "$exit_code"
