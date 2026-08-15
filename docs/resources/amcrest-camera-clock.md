# Amcrest camera clock — hardware notes

Measured 2026-08-02 against the office and pantry cameras. `_sync_camera_clock`
in `src/cat_watcher/poller.py` acts on these facts. Camera NTP stays off.

## Timezone codes

- `NTP.TimeZone=26` gives UTC-06:00. `NTP.TimeZone=24` gives UTC-04:00. The API
  doc publishes no code table. One web-UI change on hardware gave this mapping.
- `NTP.TimeZoneDesc` read `Beijing, Chongqing, Hong Kong, Urumqi` at code 26 and
  again at code 24. Do not trust the description.
- `NTP.UpdatePeriod=10` means ten minutes. The API doc gives no unit.

## setCurrentTime

These encodings of the `time` value all moved the clock:

- Literal colons with a `%20` space.
- Percent-encoded colons.
- The unpadded form from the API doc.

If a write does not hold, the encoding is not the cause.

One `setCurrentTime` call returned `OK` and did not move the clock. The cause is
unknown. The poller reads the clock back after every write for this reason.

## Threshold

`clock_drift_threshold_seconds = 60` keeps clip timestamps accurate to under a
minute without a device write on every tick.

## Baseline before the fix

From the poller logs, 2026-07-18 to 2026-08-02:

- Office drift -7198.3 s. Pantry drift -5.3 s.
- Office ingested 1 clip on a normal tick and 152 on safety-net ticks. Pantry
  ingested 213 and 20. Pantry recorded more clips, so volume does not explain
  the contrast. This ratio is the acceptance measurement.
- Median tick gap 308 s against a configured 300 s. Clock exposure after a power
  cycle is one poll interval.

## Open questions

- Clock retention across a power cycle is never tested. The operator unplugs the
  office camera every other week to clean it.
- The `Locales` DST fields carry `Year` values from 2000 to 2038. Annual
  recurrence is unverified. This is moot while NTP stays off.
- Clips stored before the fix keep wrong timestamps. The true offset for a past
  clip is not recoverable.
