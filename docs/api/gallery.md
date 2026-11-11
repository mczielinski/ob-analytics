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

::: ob_analytics.visualization.gallery.display_result

## Gallery model

::: ob_analytics.visualization.gallery.build_gallery_model

::: ob_analytics.visualization.gallery.GalleryModel

::: ob_analytics.visualization.gallery.PlotConcept

::: ob_analytics.visualization.gallery.PlotSpec

::: ob_analytics.visualization.gallery.generate_gallery

## Extra cards

The built-in [metrics](metrics.md) (the L1 quote, book signals, VPIN, Kyle's
λ, order flow imbalance, the OFI horizon graph, transaction costs, Amihud
illiquidity and the Roll spread) already have cards, with their default
settings. To add bars, LOBSTER trading halts, or a metric's plot from a frame
you computed with other settings, build a card and append it to
`model.analytics` before rendering. The `*_card` helpers below build the
common ones. For any other level-less plot, build a `PlotSpec` with the
plot's `prepare` function:

```python
from ob_analytics import Pipeline, vpin, sample_csv_path
from ob_analytics.visualization import prepare
from ob_analytics.visualization.gallery import (
    PlotSpec, build_gallery_model, generate_gallery,
)

result = Pipeline().run(sample_csv_path())
vpin_df = vpin(result.trades)

model = build_gallery_model(result)
model.analytics.append(
    PlotSpec(
        name="vpin_alert_08",
        title="VPIN (alert at 0.8)",
        plot_name="vpin",
        prepare=prepare.vpin,
        prep_kwargs={"vpin_df": vpin_df, "threshold": 0.8},
    )
)
generate_gallery(result, "out/gallery", model=model)
```

::: ob_analytics.visualization.gallery.bars_card

::: ob_analytics.visualization.gallery.ofi_by_window_card

::: ob_analytics.visualization.gallery.kyle_lambda_card

::: ob_analytics.visualization.gallery.trading_halts_card
