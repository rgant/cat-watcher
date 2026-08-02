#!/usr/bin/env bash
# Install (or refresh) the four cat-watcher LaunchAgents on this Mac.
#
# Usage:
#   install_launchagents.sh           render plists, stop every agent, then start every agent
#   install_launchagents.sh --stop    bootout every agent and leave them down
#   install_launchagents.sh --start   render plists and bootstrap every agent
#
# The split lets a deploy migrate the database with every agent stopped, instead of migrating
# underneath running ones.
#
# Two behaviours here exist because of a launchd race. `launchctl bootstrap` fails with
# `Bootstrap failed: 5: Input/output error` (EIO) when the agent's previous instance has not
# finished unwinding, and launchd reports the label gone long before that is true.
#
#   1. Every agent is booted out before any is bootstrapped, so each one gets the longest
#      possible head start rather than being re-bootstrapped a millisecond after its own bootout.
#   2. A failed bootstrap is retried, then recorded and reported rather than aborting the run.
#      Aborting used to leave later agents silently down with no indication of which.

set -euo pipefail

AGENTS=(poller alerts web backup)
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAUNCHAGENTS_DIR="${HOME}/Library/LaunchAgents"
UID_NUM="$(id -u)"
: "${BOOTSTRAP_ATTEMPTS:=5}"
: "${BOOTSTRAP_RETRY_SECONDS:=3}"

plist_for() { echo "${LAUNCHAGENTS_DIR}/com.robgant.cat-watcher.${1}.plist"; }
target_for() { echo "gui/${UID_NUM}/com.robgant.cat-watcher.${1}"; }

render_plists() {
	mkdir -p "${LAUNCHAGENTS_DIR}"
	# render_plists creates internal_root/logs/ as a side effect, so no mkdir is needed for it.
	(
		cd "${REPO_DIR}" \
			&& pixi run python -m cat_watcher.scripts.render_plists --output "${LAUNCHAGENTS_DIR}"
	)
}

stop_agents() {
	for agent in "${AGENTS[@]}"; do
		# bootout failures are expected when the agent is not loaded; swallow them.
		launchctl bootout "$(target_for "${agent}")" 2>/dev/null || true
		echo "stopped ${agent}"
	done
}

# Bootstrap one agent, retrying the EIO race. Returns non-zero when every attempt failed.
start_agent() {
	local agent="$1" attempt
	for ((attempt = 1; attempt <= BOOTSTRAP_ATTEMPTS; attempt++)); do
		if launchctl bootstrap "gui/${UID_NUM}" "$(plist_for "${agent}")" 2>/dev/null; then
			echo "loaded ${agent}"
			return 0
		fi
		if ((attempt < BOOTSTRAP_ATTEMPTS)); then
			echo "  ${agent}: bootstrap attempt ${attempt}/${BOOTSTRAP_ATTEMPTS} failed, retrying in ${BOOTSTRAP_RETRY_SECONDS}s"
			sleep "${BOOTSTRAP_RETRY_SECONDS}"
		fi
	done
	return 1
}

start_agents() {
	local failed=()
	for agent in "${AGENTS[@]}"; do
		start_agent "${agent}" || failed+=("${agent}")
	done

	if ((${#failed[@]} == 0)); then
		echo "all agents loaded: ${AGENTS[*]}"
		return 0
	fi

	echo "FAILED to load: ${failed[*]}" >&2
	echo "Recover each one with a bare bootstrap, which cannot race because it is already down:" >&2
	for agent in "${failed[@]}"; do
		echo "  launchctl bootstrap gui/${UID_NUM} $(plist_for "${agent}")" >&2
	done
	echo "Do not rerun this script to recover — its bootout would knock down a working agent." >&2
	return 1
}

case "${1-}" in
--stop)
	stop_agents
	;;
--start)
	render_plists
	start_agents
	;;
"")
	render_plists
	stop_agents
	start_agents
	;;
*)
	echo "usage: $(basename "$0") [--stop | --start]" >&2
	exit 64
	;;
esac
