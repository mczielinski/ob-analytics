"""Resolution-level taxonomy for order-book visualizations.

Every leveled plot concept declares the order-book resolution it renders at.
The level is a *coordinate* in the renderer-registry key
``(concept, level, backend)`` -- never a name suffix -- so that L2 and L3 are
symmetric variants of one concept rather than separately-named plots.
"""

from __future__ import annotations

from enum import Enum


class Level(str, Enum):
    """Order-book resolution level a plot renders at.

    ``L2`` -- Market-By-Price (MBP): aggregate volume per price level, with no
    persistent order identity.  ``L3`` -- Market-By-Order (MBO): one primitive
    per resting order, with stable identity (queue position recoverable).

    The ``str`` mixin lets members slot directly into registry tuple keys and,
    via the :meth:`__str__` override, render as the bare token (``"L2"``) in
    file stems and f-strings rather than ``"Level.L2"``.
    """

    L2 = "L2"
    L3 = "L3"

    def __str__(self) -> str:  # noqa: D105 -- bare token for file stems / f-strings
        return self.value
