---
title: Capture record
---

# Capture record

Every live capture, and every segment of a long one, writes a `meta.json`
beside its files. Most of its fields are counters. A few say how to read the
data back: the source that made the capture and what it declares about its
feed, and the instrument's tick size and lot size. This module writes and reads
those fields. It is outside `ob_analytics.live`, so code that only replays files
does not need the live-capture extras.

`Pipeline.run` reads the record of the capture it is given, and uses the
recorded tick size and lot size when your config does not set them. So
`Pipeline.from_source("depth_csv").run(capture)` and
`ob-analytics process capture --source depth_csv` give the same result. See
[Capture live data](../howto/live-capture.md).

::: ob_analytics.capture_record.read_record

::: ob_analytics.capture_record.CaptureRecord

::: ob_analytics.capture_record.read_instrument

::: ob_analytics.capture_record.write_record

::: ob_analytics.capture_record.source_declarations

::: ob_analytics.capture_record.read_fields

::: ob_analytics.capture_record.record_path
