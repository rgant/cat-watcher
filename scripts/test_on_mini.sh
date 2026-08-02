#!/usr/bin/env bash
# Run the test suite against the code currently deployed on media-center.
#
# This leaves the mini's checkout alone: a test task that changes what is deployed cannot tell you
# whether what is deployed works. Deploying is `pixi run deploy-update`, run on the mini itself.
#
# The remote command goes through `bash -lc` because `ssh host 'cmd'` runs a non-login shell whose
# PATH is only /usr/bin:/bin:/usr/sbin:/sbin. Homebrew installs pixi outside that PATH, so a login
# shell is what makes it resolvable.

set -euo pipefail

ssh media-center "bash -lc 'cd ~/Programming/cat-watcher && pixi run pytest'"
