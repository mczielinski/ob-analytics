---
title: Matching Engine
---

# Matching Engine

Pairs bid/ask fills to identify which events are part of the same trade.
The default Bitstamp matcher uses Needleman--Wunsch sequence alignment;
LOBSTER uses a pass-through (single-sided executions need no pairing).

::: ob_analytics.matching_engine.NeedlemanWunschMatcher
