---
title: Gallery
---

# Gallery

HTML gallery generation from pipeline results. Produces a self-contained
page with all standard plots for quick visual inspection.

The gallery is built from a list of [`PlotSpec`](#ob_analytics.visualization.gallery.PlotSpec)
entries. Pass `specs=` to override the defaults, or call
[`default_specs`](#ob_analytics.visualization.gallery.default_specs) to extend
them.

::: ob_analytics.visualization.gallery.generate_gallery

::: ob_analytics.visualization.gallery.PlotSpec

::: ob_analytics.visualization.gallery.default_specs

## Extra panels

Build optional flow-toxicity and LOBSTER panels and append them via
`generate_gallery(..., extra_panels=[...])`.

::: ob_analytics.visualization.gallery.vpin_panel

::: ob_analytics.visualization.gallery.ofi_panel

::: ob_analytics.visualization.gallery.kyle_panel

::: ob_analytics.visualization.gallery.trading_halts_panel
