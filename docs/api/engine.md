---
title: Order-book engine
---

# Order-book engine

The stateful core: **order events in, book states and order lifecycles out**.
This is the only part of the library that knows how a book is rebuilt, and the
only part that holds no pandas — everything crosses its interface as NumPy
arrays, so the inside can be made faster later (numba, then possibly Rust)
without anything above it moving.

Most users never call it directly. Its frame-level faces are
[`order_book`](analytics.md#ob_analytics.analytics.order_book),
[`order_lifecycles`](analytics.md#ob_analytics.analytics.order_lifecycles),
and
the functions in `ob_analytics.queue`; they convert to and from these arrays.

## What crosses the interface

`OrderEvents` is the shared event schema ([data contracts](schemas.md)) in
column form. Results carry a **row index** back into those arrays rather than
copying the event's own columns out, so a caller reads the
`exchange_timestamp`, the classifier `type`, the venue and symbol, or anything
else it tracks off its own table at that row. Adding a column to the schema
therefore never widens this interface, and the engine never has to learn a
vocabulary — order types, venue names — that belongs to the layer above.

Timestamps are **int64 nanoseconds since the epoch, UTC**: a time zone is a
presentation detail, so it is stripped on the way in and re-attached on the way
out. Prices are **integer ticks**; the engine only compares and subtracts them,
so a float column from a pre-tick frame still works.

Categorical columns arrive as integer codes — `Direction`, `Action`, `Outcome`.
Each derives the schema's string from its own member name, so the integer the
engine compares and the label a frame carries cannot fall out of step.

## Input

::: ob_analytics.engine.OrderEvents

## Codes

::: ob_analytics.engine.Direction

::: ob_analytics.engine.Action

::: ob_analytics.engine.Outcome

## Reconstructions

::: ob_analytics.engine.book_state

::: ob_analytics.engine.order_lifecycles

::: ob_analytics.engine.queue_positions

::: ob_analytics.engine.queue_age_grid

::: ob_analytics.engine.crossed_prefix_counts

## Results

::: ob_analytics.engine.BookState

::: ob_analytics.engine.BookSide

::: ob_analytics.engine.OrderLifecycles

::: ob_analytics.engine.QueuePositions

::: ob_analytics.engine.QueueAgeGrid
