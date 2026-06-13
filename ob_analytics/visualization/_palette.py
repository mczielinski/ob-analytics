"""Semantic color constants shared by every rendering backend.

Three categorical vocabularies, one hue pair/triple each:

* **Side** (resting order's side): bid / ask — book snapshot, depth chart,
  cancellations, best-quote lines.
* **Fate** (order lifecycle outcome): flashed (cancelled) / resting (filled),
  plus the competing-risks split filled / partial / cancelled.
* **Aggressor** (taker side of an execution): buy / sell — trade tape and
  trade markers.

Backends import these so the matplotlib and plotly faces stay identical.

The hues derive from the Okabe–Ito colorblind-safe palette and were verified
under Machado-matrix deuteranopia simulation (the most common color-vision
deficiency): every pair keeps a normalized RGB distance ≥ 0.26 (dominant
pairs ≥ 0.5), where the previous buy-green/sell-red pair collapsed to 0.13.
Luminances are spread (0.37–0.64) so the classes also survive grayscale.
"""

# Side (bid / ask) — Okabe–Ito blue / vermillion
_BID_COLOR = "#0072B2"
_ASK_COLOR = "#D55E00"

# Fate: order_activity L3 Gantt
_FLASHED_COLOR = "#E69F00"  # flashed-limit: placed and pulled (cancelled)
_RESTING_COLOR = "#009E73"  # resting-limit: rested / filled

# Fate: order_outcome L3 competing-risks scatter
_FILLED_COLOR = "#009E73"  # fully executed (bluish green)
_PARTIAL_COLOR = "#CC79A7"  # partially executed, remainder removed
_CANCELLED_COLOR = "#E69F00"  # removed without any execution (orange)

# Aggressor (taker side) — bluish green / vermillion
_BUY_COLOR = "#009E73"  # buyer-initiated execution (lifts the ask)
_SELL_COLOR = "#D55E00"  # seller-initiated execution (hits the bid)
