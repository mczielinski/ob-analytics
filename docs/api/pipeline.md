---
title: Pipeline
---

# Pipeline

The main orchestrator. Runs the full sequence: load → build trades →
classify → depth → metrics → (optional VPIN/OFI). Use `Pipeline(format=...)`
for LOBSTER or other registered formats, or pass individual components
(`loader=`, `trade_source=`) to override specific stages.

::: ob_analytics.pipeline.Pipeline

::: ob_analytics.pipeline.PipelineResult

## Format registry

::: ob_analytics.pipeline.register_format

::: ob_analytics.pipeline.list_formats
