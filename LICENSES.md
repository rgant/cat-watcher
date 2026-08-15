# Third-Party Notices

This project is licensed under the AGPL-3.0-or-later (see `LICENSE`).

## Bundled / referenced third-party works

### ultralytics (runtime dependency)

Licensed under the AGPL-3.0. <https://github.com/ultralytics/ultralytics>

### YOLOv11n model weights

Distributed by Ultralytics under the AGPL-3.0. Downloaded at first run into
`models/yolo11n.pt`.

### htmx 2.0.4

Licensed under the Zero-Clause BSD (0BSD).
<https://github.com/bigskysoftware/htmx> The `htmx.org` npm package pins the
version. `scripts/vendor_static_assets.sh` copies `dist/htmx.min.js` verbatim to
`src/cat_watcher/web/static/vendor/htmx.min.js`, which the web app serves.

### Amcrest HTTP API documentation

`docs/resources/Amcrest-HTTP_API_V3.26.pdf` — vendor reference document, used
only for development. Not redistributed.
