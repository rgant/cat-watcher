#!/usr/bin/env bash
# Install (or refresh) the four cat-watcher LaunchAgents on this Mac.
#
# Idempotent: a `bootout` of any already-loaded agent precedes `bootstrap`, so re-running after
# edits to config.toml or the plist templates cleanly picks up the new values.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHAGENTS_DIR="${HOME}/Library/LaunchAgents"
UID_NUM="$(id -u)"

mkdir -p "${LAUNCHAGENTS_DIR}"

# render_plists creates internal_root/logs/ as a side effect, so no mkdir is needed here.
(
	cd "${REPO_DIR}" \
		&& pixi run python -m cat_watcher.scripts.render_plists --output "${LAUNCHAGENTS_DIR}"
)

for agent in poller alerts web backup; do
	plist="${LAUNCHAGENTS_DIR}/com.robgant.cat-watcher.${agent}.plist"
	target="gui/${UID_NUM}/com.robgant.cat-watcher.${agent}"

	# bootout failures are expected on first install (nothing to remove); swallow them.
	launchctl bootout "${target}" 2>/dev/null || true
	launchctl bootstrap "gui/${UID_NUM}" "${plist}"
	echo "loaded ${agent}"
done
