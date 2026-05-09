#!/usr/bin/env bash
# Remove the four cat-watcher LaunchAgents from this Mac. The plist templates under scripts/plists/
# are left untouched.

set -euo pipefail

LAUNCHAGENTS_DIR="${HOME}/Library/LaunchAgents"
UID_NUM="$(id -u)"

for agent in poller alerts web backup; do
	target="gui/${UID_NUM}/com.robgant.cat-watcher.${agent}"
	plist="${LAUNCHAGENTS_DIR}/com.robgant.cat-watcher.${agent}.plist"

	launchctl bootout "${target}" 2>/dev/null || true
	rm -f "${plist}"
	echo "removed ${agent}"
done
