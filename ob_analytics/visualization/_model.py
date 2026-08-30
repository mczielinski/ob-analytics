"""Resolution-level taxonomy for order-book visualizations.

Every leveled plot concept declares the order-book resolution it renders at.
The level is a *coordinate* in the renderer-registry key
``(concept, level, backend)`` -- never a name suffix -- so that L2 and L3 are
symmetric variants of one concept rather than separately-named plots.

:class:`Level` is defined in :mod:`ob_analytics.protocols` (it is also the
axis a :class:`~ob_analytics.protocols.Source` declares its ingestion
level on, via ``Source.level``) and re-exported here so the visualization
layer keeps a local name.
"""

from __future__ import annotations

from ob_analytics.protocols import Level

__all__ = ["Level"]
