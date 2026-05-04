# Third-Party Notices

This project is licensed under the AGPL-3.0-or-later (see `LICENSE`).

## Bundled / referenced third-party works

### ultralytics (runtime dependency)

Licensed under the AGPL-3.0. <https://github.com/ultralytics/ultralytics>

### YOLOv11n model weights

Distributed by Ultralytics under the AGPL-3.0. Downloaded at first run into
`models/yolov11n.pt`.

### Test fixture: `tests/fixtures/cat_image.jpg`

Photograph of a calico kitten (COCO class 15 = "cat"), used by the detector test
suite. Bundled in this repository so the test suite is offline-reproducible.

- **Source:** Wikimedia Commons,
  [File:2008-11-28 Calico kitten on the litter box.jpg](https://commons.wikimedia.org/wiki/File:2008-11-28_Calico_kitten_on_the_litter_box.jpg)
  (uploaded by SSJF01; original photo from
  <https://www.flickr.com/photos/sfmine79/3079536095/>)
- **Author:** MiNe (Flickr user `sfmine79`)
- **License:** Creative Commons Attribution 2.0
  ([CC BY 2.0](https://creativecommons.org/licenses/by/2.0))
- **Modifications:** Scaled from 2430x1620 to 400x267 with
  `ffmpeg -vf scale=400:-1` and re-encoded with `-q:v 7` to keep the bundled
  fixture under 30 KB.
- **Attribution requirement:** Per CC BY 2.0, this notice constitutes the
  required attribution for any redistribution of the fixture image.

### Amcrest HTTP API documentation

`docs/resources/Amcrest-HTTP_API_V3.26.pdf` — vendor reference document, used
only for development. Not redistributed.
