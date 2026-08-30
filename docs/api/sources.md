---
title: Sources
---

# Source registry

Every data source — file or live — registers here under a name, the one
registry shared by both. Built-in sources self-register on import; third-party
sources load through the `ob_analytics.sources` entry-point group (see
[Extending](../extending.md#shipping-a-source-as-its-own-package)). A registered
value is a [`Source`](protocols.md) class; construct it and use the capability
you need — `OfflineSource` for file replay (`Pipeline(source=...)`), `LiveSource`
for live capture.

Per-source configuration is a typed `SourceSettings` (a frozen pydantic model),
not an untyped dict — subclass it per source, e.g. `CcxtSettings`.

::: ob_analytics.sources.register_source

::: ob_analytics.sources.list_sources

::: ob_analytics.sources.get_source

::: ob_analytics.sources.load_source_plugins

::: ob_analytics.config.SourceSettings
