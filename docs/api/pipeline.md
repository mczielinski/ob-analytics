---
title: Pipeline
---

# Pipeline

The main orchestrator. Runs the full sequence: load → build trades →
classify → depth. Use `Pipeline(format=...)` for LOBSTER or other registered
formats, or pass individual components (`loader=`, `trade_source=`) to override
specific stages. Flow-toxicity metrics are computed *after* the run by calling
`compute_vpin` / `compute_kyle_lambda` / `order_flow_imbalance` on
`result.trades`.

::: ob_analytics.pipeline.Pipeline

::: ob_analytics.pipeline.PipelineResult

## Format registry

::: ob_analytics.pipeline.register_format

::: ob_analytics.pipeline.list_formats
