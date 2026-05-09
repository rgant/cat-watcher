# cat-watcher Development Tools
# Install with: brew bundle
#
# Most lint/format tools (shellcheck, dprint, markdownlint-cli, actionlint, zizmor) live in pixi
# (pixi.lock pins them across platforms). The remaining brew entries below are tools either not
# packaged for conda-forge (shfmt) or load-bearing outside the pixi env (git, pixi itself, ffmpeg
# for ad-hoc shell use).

# Shell formatter — not on conda-forge. CI installs via `apt-get install shfmt`.
brew "shfmt"

# Media processing (clip frame sampling + thumbnails)
brew "ffmpeg"

# Git Source Control
brew "git"

# Python package management + task runner (also serves as the project's task runner via [tool.pixi.tasks])
brew "pixi"

# Node version manager — `.nvmrc` pins the project's Node version; `nvm install && nvm use` picks it up.
brew "nvm"
