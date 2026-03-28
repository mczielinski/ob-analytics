---
title: Pipeline
---

# Pipeline

The main orchestrator. Runs the full sequence: load → match → infer trades →
classify → depth → metrics → (optional VPIN/OFI). Use `Pipeline(format=...)`
for LOBSTER or other registered formats, or pass individual components
(`loader=`, `matcher=`, `trade_inferrer=`) to override specific stages.

::: ob_analytics.pipeline.register_format

::: ob_analytics.pipeline.PipelineResult

::: ob_analytics.pipeline.Pipeline
