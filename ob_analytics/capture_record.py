"""The capture record: the ``meta.json`` a live capture writes beside its files.

Every capture, and every segment of a long one, has a ``meta.json``.  Most of
its fields are counters for a person to read.  A few say how to read the data
back, and this module owns those:

* what the source that made the capture declares about its feed
  (``source``, ``feed_type``, ``trade_attribution``, ``sequence_kind``,
  ``clocks``), which ``audit`` holds the capture to;
* how many times the venue's sequence started again
  (``sequence_restarts``), which ``audit`` does not count as faults;
* the instrument's price and size steps (``tick_size``, ``lot_size``), which
  a replay needs to put prices and sizes on the right grid.

The capture writes the record with :func:`write_record`, and
:func:`read_record` reads it back as a :class:`CaptureRecord`.
:meth:`~ob_analytics.pipeline.Pipeline.run` applies the recorded tick and lot
size, so ``Pipeline(...).run(capture)`` and ``ob-analytics process capture``
read a capture the same way.

This module is outside :mod:`ob_analytics.live`, so code that only replays
files does not import the live-capture package.
"""

from __future__ import annotations

import json
import math
import os
import warnings
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeVar

from loguru import logger

from ob_analytics._secrets import redact
from ob_analytics.config import instrument_fields
from ob_analytics.protocols import (
    Clocks,
    FeedType,
    SequenceKind,
    TradeAttribution,
    clocks_of,
    trade_attribution_of,
)

#: The record's file name, in the capture folder or a segment's folder.
RECORD_NAME = "meta.json"

# The names of the fields this module reads.  A capture writes many more.
SOURCE = "source"
FEED_TYPE = "feed_type"
TRADE_ATTRIBUTION = "trade_attribution"
SEQUENCE_KIND = "sequence_kind"
CLOCKS = "clocks"
SEQUENCE_RESTARTS = "sequence_restarts"
TICK_SIZE = "tick_size"
LOT_SIZE = "lot_size"

_E = TypeVar("_E", bound=Enum)


@dataclass(frozen=True)
class CaptureRecord:
    """What a capture recorded about how to read its data.

    Every field is ``None`` (``0`` for ``sequence_restarts``) when the capture
    did not record it: input that is not a capture, a capture written before
    captures recorded the field, or a venue that does not give the value.

    Attributes
    ----------
    source : str or None
        The name of the source that made the capture.  It can differ from the
        source that reads the files: a cryptofeed L3 capture is read as
        ``bitstamp``.
    feed_type : FeedType or None
        What that source declares about crossing (see
        :class:`~ob_analytics.protocols.FeedType`).
    trade_attribution : TradeAttribution or None
        Which orders of a trade that source's feed names (see
        :class:`~ob_analytics.protocols.TradeAttribution`).
    sequence_kind : SequenceKind or None
        What the venue ``sequence`` promises (see
        :class:`~ob_analytics.protocols.SequenceKind`).  ``None`` when the
        source declares none; then use what the source declares now
        (:func:`~ob_analytics.protocols.sequence_kind_of`).
    clocks : Clocks or None
        Which clocks the capture's rows carry (see
        :class:`~ob_analytics.protocols.Clocks`).  A live source learns this
        from the venue's books, so it is written when the capture closes.
    sequence_restarts : int
        How many times the venue's sequence started again.  A source that
        finds a lost message itself starts again from a new opening book, and
        on some venues the sequence starts again too.  These steps back are
        not faults (see
        :func:`~ob_analytics.analytics.data_quality_summary`).
    tick_size : float or None
        The instrument's price step, in the quote currency.
    lot_size : float or None
        The instrument's size step, in the base asset.
    """

    source: str | None = None
    feed_type: FeedType | None = None
    trade_attribution: TradeAttribution | None = None
    sequence_kind: SequenceKind | None = None
    clocks: Clocks | None = None
    sequence_restarts: int = 0
    tick_size: float | None = None
    lot_size: float | None = None

    def instrument(self) -> dict[str, Any]:
        """Return the :class:`~ob_analytics.config.PipelineConfig` fields recorded.

        The recorded tick size comes with the ``price_decimals`` that shows
        one tick, and the lot size with its ``volume_decimals``.  A step the
        capture did not record is left out.
        """
        return instrument_fields(tick_size=self.tick_size, lot_size=self.lot_size)


def record_path(path: Any) -> Path:
    """Return where the record of the capture at *path* is.

    *path* is a capture folder, a segment's folder, a file inside one, or the
    output of ``ob-analytics process``, which keeps the capture's record.
    """
    p = Path(path)
    return (p.parent if p.is_file() else p) / RECORD_NAME


def read_fields(path: Any) -> dict[str, Any] | None:
    """Return every field of the record at *path*, as written.

    ``None`` when there is no record, or it is not a JSON object.  Any other
    error reading the file is raised, so a writer never replaces a record it
    could not read.
    """
    try:
        data = json.loads(record_path(path).read_text())
    except (FileNotFoundError, NotADirectoryError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def read_record(path: Any) -> CaptureRecord:
    """Read the record of the capture at *path*.

    Parameters
    ----------
    path
        A capture folder, a segment's folder, a file inside one, or the output
        of ``ob-analytics process``.  Anything that is not a path (a frame,
        say) has no record.

    Returns
    -------
    CaptureRecord
        What the capture recorded.  Empty when there is no ``meta.json``, or
        it cannot be read.  A declaration with a value this version does not
        know is left out with a warning.
    """
    if not isinstance(path, str | os.PathLike):
        return CaptureRecord()
    fields = _readable_fields(path)
    where = record_path(path)
    source = fields.get(SOURCE)
    return CaptureRecord(
        source=str(source) if source else None,
        feed_type=_enum(fields, FEED_TYPE, FeedType, where),
        trade_attribution=_enum(fields, TRADE_ATTRIBUTION, TradeAttribution, where),
        sequence_kind=_enum(fields, SEQUENCE_KIND, SequenceKind, where),
        clocks=_enum(fields, CLOCKS, Clocks, where),
        sequence_restarts=_count(fields, SEQUENCE_RESTARTS),
        tick_size=_step(fields, TICK_SIZE),
        lot_size=_step(fields, LOT_SIZE),
    )


def read_instrument(path: Any) -> dict[str, Any]:
    """Return the :class:`~ob_analytics.config.PipelineConfig` fields a capture recorded.

    The same as ``read_record(path).instrument()``, without reading the
    source's declarations, so a declaration this version does not know is not
    warned about.  This is what :meth:`~ob_analytics.pipeline.Pipeline.run`
    reads.

    Parameters
    ----------
    path
        As for :func:`read_record`.

    Returns
    -------
    dict of str to Any
        The recorded ``tick_size`` and ``lot_size``, each with its display
        precision; empty when the capture records neither.
    """
    if not isinstance(path, str | os.PathLike):
        return {}
    fields = _readable_fields(path)
    return instrument_fields(
        tick_size=_step(fields, TICK_SIZE), lot_size=_step(fields, LOT_SIZE)
    )


def _enum(fields: dict[str, Any], key: str, kind: type[_E], where: Path) -> _E | None:
    """The declaration *key* as a *kind*, or ``None`` when absent or unknown."""
    value = fields.get(key)
    if not value:
        return None
    try:
        return kind(value)
    except ValueError:
        warnings.warn(
            f"{where}: {key} {value!r} is not a value this version knows "
            f"({', '.join(str(m.value) for m in kind)}); it is ignored.",
            UserWarning,
            stacklevel=3,
        )
        return None


def _readable_fields(path: Any) -> dict[str, Any]:
    """The record's fields for a reader: empty when there is none or it fails."""
    try:
        return read_fields(path) or {}
    except OSError as exc:
        warnings.warn(
            f"{record_path(path)} cannot be read ({exc}); it is ignored.",
            UserWarning,
            stacklevel=4,
        )
        return {}


def _count(fields: dict[str, Any], key: str) -> int:
    """A recorded count, or ``0`` when absent or not a whole number from 0 up."""
    value = fields.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0


def _step(fields: dict[str, Any], key: str) -> float | None:
    """A recorded tick or lot size, or ``None`` when absent or not a step.

    A step is a finite number above zero; ``true`` or ``"inf"`` is not one.
    """
    value = fields.get(key)
    if isinstance(value, bool):
        return None
    try:
        step = float(value) if value else None
    except (TypeError, ValueError):
        return None
    return step if step is not None and math.isfinite(step) and step > 0 else None


def write_record(folder: Path, fields: dict[str, Any]) -> None:
    """Write *fields* as the record of the capture in *folder*.

    The file is replaced in one step, so a reader never sees half of it, and
    any API key the capture holds is removed from the text first.  A value
    JSON cannot hold is written as its text.
    """
    path = Path(folder) / RECORD_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(redact(json.dumps(fields, indent=2, default=str)))
    os.replace(tmp, path)


def source_declarations(source: Any) -> dict[str, Any]:
    """Return what a live *source* declares about its feed, as record fields.

    ``source`` names the source; ``feed_type`` and ``trade_attribution`` are
    what it declares (see :class:`~ob_analytics.protocols.FeedType` and
    :class:`~ob_analytics.protocols.TradeAttribution`).  ``sequence_kind`` is
    written only by a source that declares one.  ``clocks`` (see
    :class:`~ob_analytics.protocols.Clocks`) is read when the capture closes,
    since a live source learns it from the venue's books.

    It runs while a capture is closing, so a declaration that cannot be read
    (a plug-in's value outside the enum, say) is logged and never stops the
    files being finished: an unreadable ``feed_type`` is recorded as
    ``unknown``, and an unreadable ``trade_attribution`` is left out, so
    ``audit`` falls back to ``--source`` rather than to a guess.
    """
    declared: dict[str, Any] = {SOURCE: source.name}
    try:
        declared[FEED_TYPE] = FeedType(
            getattr(source, "feed_type", FeedType.UNKNOWN)
        ).value
    except Exception as exc:  # noqa: BLE001 - never block finalize
        logger.warning("Capturer '{}' feed_type unreadable: {!r}", source.name, exc)
        declared[FEED_TYPE] = FeedType.UNKNOWN.value
    try:
        declared[TRADE_ATTRIBUTION] = trade_attribution_of(source).value
    except Exception as exc:  # noqa: BLE001 - never block finalize
        logger.warning(
            "Capturer '{}' trade_attribution unreadable: {!r}", source.name, exc
        )
    sequence_kind = getattr(source, "sequence_kind", None)
    if sequence_kind is not None:
        declared[SEQUENCE_KIND] = str(getattr(sequence_kind, "value", sequence_kind))
    try:
        declared[CLOCKS] = clocks_of(source).value
    except Exception as exc:  # noqa: BLE001 - never block finalize
        logger.warning("Capturer '{}' clocks unreadable: {!r}", source.name, exc)
    return declared
