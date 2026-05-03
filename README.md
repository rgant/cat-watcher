# cat-watcher

Monitor indoor litter-box cameras: ingest motion clips, classify cat-vs-no-cat,
browse via web UI, and alert on inactivity / unusual frequency / agent failures.

## Quick start (development)

```bash
brew bundle
pixi install
cp config.example.toml config.toml
cp .env.example .env  # fill in real secrets
mkdir data
pixi run db-upgrade
pixi run dev
```

`brew bundle` installs system tools (including pixi itself); `pixi install` then
provisions the Python/conda environment. Both are required and must run in that
order.

## Useful commands

Run `pixi task list` to see all configured tasks. A few common operations aren't
wrapped as tasks — invoke their underlying binaries directly via pixi:

```bash
pixi run cat-watcher status              # show service health
pixi run cat-watcher test-cameras        # verify camera connectivity
pixi run cat-watcher test-notification   # send a test alert
pixi run cat-watcher-backup              # back up the database
pixi run markdownlint --fix .            # lint / auto-fix Markdown
pixi tree                                # dependency tree
```
