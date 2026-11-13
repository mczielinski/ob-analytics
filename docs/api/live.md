---
title: Live capture
---

# Live capture

Run a live source into a directory of segments, and read back the
`manifest.json` that records them. See
[Capture live data](../howto/live-capture.md#running-for-days) for how
segments, rolls, reconnects and restarts work.

::: ob_analytics.live.run_capture

::: ob_analytics.live.CaptureRun

::: ob_analytics.live.CaptureConfig

::: ob_analytics.live.read_manifest

::: ob_analytics.live.CaptureManifest

::: ob_analytics.live.Segment

::: ob_analytics.live.Gap

::: ob_analytics.live.EndReason

## Built-in live sources

The `bitstamp` source captures Bitstamp itself, and the `kalshi` source
captures Kalshi from its WebSocket feed (it needs the `kalshi` extra and an API
key). The `ccxt` and `cryptofeed` sources reach many venues through those
libraries, and need the `ccxt` and `cryptofeed` extras. See
[Capture Kalshi](../howto/kalshi.md#capture-every-change-with-an-api-key),
[Capture CCXT venues](../howto/ccxt.md) and
[Capture cryptofeed venues](../howto/cryptofeed.md).

::: ob_analytics.live.bitstamp.BitstampCapturer

::: ob_analytics.live.CcxtSource

::: ob_analytics.live.CcxtSettings

::: ob_analytics.live.CryptofeedSource

::: ob_analytics.live.CryptofeedSettings

::: ob_analytics.live.KalshiSource

::: ob_analytics.live.KalshiSettings

## Writing a live source

What a live source yields and what the capture runner reads from it. See
[The live capability](../extending.md#the-live-capability).

::: ob_analytics.live.EventDict

::: ob_analytics.live.CaptureSink

::: ob_analytics.live.CaptureResult

::: ob_analytics.live.SupportsDiagnostics

::: ob_analytics.live.SupportsPreflight
