#!/usr/bin/env bash
# Copy browser-side npm packages into the directory the web app serves.
#
# The static mount reads src/cat_watcher/web/static/, so a file under node_modules/ never reaches a
# browser. The copy is committed, which keeps the UI working on a checkout that never ran npm ci.
# package-lock.json stays the one place a version is pinned.
#
# Run with --check to compare instead of copy. The ``vendor-static-assets`` pre-commit hook uses
# that mode, so a version bump without a re-copy fails before it lands.

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

dest_dir="src/cat_watcher/web/static/vendor"

# One entry per vendored asset: "<path under node_modules> <filename under $dest_dir>".
assets=(
	"htmx.org/dist/htmx.min.js htmx.min.js"
)

check_only=false
if [ "${1:-}" = "--check" ]; then
	check_only=true
fi

exit_code=0
for entry in "${assets[@]}"; do
	read -r source_rel dest_name <<<"$entry"
	source="node_modules/$source_rel"
	dest="$dest_dir/$dest_name"

	if [ ! -f "$source" ]; then
		echo "missing $source — run 'npm ci' first" >&2
		exit_code=1
		continue
	fi

	if [ "$check_only" = true ]; then
		if ! cmp --silent "$source" "$dest"; then
			echo "$dest differs from $source — run 'pixi run vendor-static-assets'" >&2
			exit_code=1
		fi
		continue
	fi

	mkdir -p "$dest_dir"
	cp "$source" "$dest"
	echo "vendored $dest"
done
exit "$exit_code"
