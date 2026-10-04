---
title: Use an API key
---

# Use an API key

Most sources read public data and need no key. A source that reads data only
account holders can see declares the keys it needs, and you give them to it in
environment variables. A key is never a command-line option, because the shell
keeps every command line in its history.

## Find out which keys a source needs

`ob-analytics sources` lists each source. A source that needs a key ends its
line with the environment variables to set:

```text
<source>  [live, key: <VENUE>_API_KEY, <VENUE>_PRIVATE_KEY_PATH]
```

No built-in source needs a key yet. The [venues](#venues) section lists those
planned.

## Set the key

Set each variable the source names before you run `capture`. Do not type the
key into the command itself, because that also goes into the shell history.
Keep it in a file and read the file:

```bash
chmod 600 ~/.config/venue/api-key
export VENUE_API_KEY="$(cat ~/.config/venue/api-key)"
```

Some venues sign each connection with a private key that lives in a file. For
those, the variable holds the path of the file, not the key:

```bash
chmod 600 ~/.config/venue/private-key.pem
export VENUE_PRIVATE_KEY_PATH=~/.config/venue/private-key.pem
```

If a key is missing, the capture stops before it creates the output directory.
The error names each variable to set and where the venue issues keys. A key
shorter than eight characters is refused, since no venue issues one that short:
use the real key, not a placeholder such as `test`.

`capture` reads the key once, when it starts. A long capture keeps running if
you move or replace the key file while it runs.

## Give the key from Python

A source's settings read an unset key from its environment variable. To give
the key yourself, pass it to the settings. A value you pass is used instead of
the environment. For a key file, pass a `Path` and the settings read the file:

```python
from pathlib import Path

from pydantic import SecretStr

settings = VenueSettings(
    api_key=SecretStr(key),  # key: a str from wherever you keep it
    private_key=Path("~/.config/venue/private-key.pem"),
)
```

Build the settings once and give the same settings to every segment of a
capture. A factory that builds new settings reads the key again for each
segment, and stops the capture if the key file has gone:

```python
from ob_analytics.live import run_capture

run = await run_capture(lambda: VenueSource(settings=settings), config)
```

## What ob-analytics does with a key

- The settings hold the key as a pydantic `SecretStr`. It prints as
  `**********` in a `repr`, a log line that shows the settings, and
  `model_dump`.
- No file a capture writes holds the key: `orders.csv`, `depth.csv`,
  `trades.csv`, `raw.jsonl`, `meta.json` and `manifest.json`. If a venue sends
  the key back in a message, or an error message quotes it, the file has
  `**********` where the key was.
- The command line's log does not hold the key either. From Python, the files
  are cleaned the same way, but a log line that a source writes itself is
  cleaned only when the command line set up the logging.

ob-analytics cannot protect the environment of the process. Other programs that
run as your user can read it.

## Venues

No built-in venue needs a key yet. These are planned, and each will add a
section here:

- Kalshi's WebSocket feed
  ([#240](https://github.com/mczielinski/ob-analytics/issues/240)). Kalshi's
  REST capture needs no key: see [Capture Kalshi](kalshi.md).
- Coinbase's per-order (L3) feed
  ([#99](https://github.com/mczielinski/ob-analytics/issues/99)). Coinbase's
  price-level (L2) book needs no key: see [Capture Coinbase](coinbase.md).

## Write a source that needs a key

[Extending ob-analytics](../extending.md#api-keys) shows how a source declares
its keys.
