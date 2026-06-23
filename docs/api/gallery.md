---
title: Gallery
---

# Gallery

HTML gallery generation from pipeline results, plus the one-line
[`plot_result`](#ob_analytics.visualization.gallery.plot_result) entry point for
rendering a single concept straight from a [`PipelineResult`](pipeline.md).

The gallery is assembled from a
[`GalleryModel`](#ob_analytics.visualization.gallery.GalleryModel) of
[`PlotConcept`](#ob_analytics.visualization.gallery.PlotConcept) entries built by
[`build_gallery_model`](#ob_analytics.visualization.gallery.build_gallery_model);
[`generate_gallery`](#ob_analytics.visualization.gallery.generate_gallery) renders
it to a self-contained HTML page.

## One-line plotting

::: ob_analytics.visualization.gallery.plot_result

::: ob_analytics.visualization.gallery.available_concepts

## Gallery model

::: ob_analytics.visualization.gallery.build_gallery_model

::: ob_analytics.visualization.gallery.GalleryModel

::: ob_analytics.visualization.gallery.PlotConcept

::: ob_analytics.visualization.gallery.PlotSpec

::: ob_analytics.visualization.gallery.generate_gallery

## Extra panels

Build optional flow-toxicity and LOBSTER panels and append them via
`generate_gallery(..., extra_panels=[...])`.

::: ob_analytics.visualization.gallery.vpin_panel

::: ob_analytics.visualization.gallery.ofi_panel

::: ob_analytics.visualization.gallery.ofi_horizon_panel

::: ob_analytics.visualization.gallery.kyle_panel

::: ob_analytics.visualization.gallery.trading_halts_panel
