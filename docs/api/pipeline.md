---
title: Pipeline
---

# Pipeline

The main orchestrator. Runs the full sequence: load → build trades →
classify → depth. Use `Pipeline(source=...)` for LOBSTER or other registered
sources, or pass individual components (`loader=`, `trade_source=`) to override
specific stages. Flow-toxicity metrics are computed *after* the run by calling
`compute_vpin` / `compute_kyle_lambda` / `order_flow_imbalance` on
`result.trades`.

The source registry (`register_source` / `list_sources` / `get_source`) lives
in [`ob_analytics.sources`](sources.md).

::: ob_analytics.pipeline.Pipeline

::: ob_analytics.pipeline.PipelineResult
