"""Semantic color constants shared by every rendering backend.

Three categorical vocabularies, one hue pair/triple each:

* **Side** (resting order's side): bid / ask — book snapshot, depth chart,
  cancellations.
* **Fate** (order lifecycle outcome): flashed (cancelled) / resting (filled),
  plus the competing-risks split filled / partial / cancelled.
* **Aggressor** (taker side of an execution): buy / sell — trade tape.

Backends import these so the matplotlib and plotly faces stay identical.
A colorblind-safe re-pick of the pairs is planned — see
``docs/plans/2026-06-10-implementation-roadmap.md`` §3.0.
"""

# Side (bid / ask)
_BID_COLOR = "#4477dd"
_ASK_COLOR = "#dd4444"

# Fate: order_activity L3 Gantt
_FLASHED_COLOR = "#e09f3e"  # flashed-limit: placed and pulled (cancelled)
_RESTING_COLOR = "#2a9d8f"  # resting-limit: rested / filled

# Fate: order_outcome L3 competing-risks scatter
_FILLED_COLOR = "#2a9d8f"  # fully executed
_PARTIAL_COLOR = "#8c8cd8"  # partially executed, remainder removed
_CANCELLED_COLOR = "#e09f3e"  # removed without any execution

# Aggressor (taker side)
_BUY_COLOR = "#2e9e5b"  # buyer-initiated execution (lifts the ask)
_SELL_COLOR = "#dd4444"  # seller-initiated execution (hits the bid)
