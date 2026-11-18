"""The record a data-quality check reports, and how much a failure matters.

:mod:`ob_analytics.quality` publishes :class:`QualityCheck` and
:class:`Severity`.  They live here, in the shared layer, so that sources can
report their own checks, such as a capture manifest's, without importing the
analytics layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

__all__ = ["QualityCheck", "Severity"]


class Severity(str, Enum):
    """How much a failed data-quality check matters.

    A check carries its severity so the policy — what fails a run — lives with
    the measurement rather than in each caller.  The enum mixes in ``str``
    (``Severity.ERROR == "error"``), which keeps CLI and JSON output plain.

    Attributes
    ----------
    ERROR
        The data contradicts something that must hold (a duplicate
        ``event_id``, a dropped venue message, a negative volume).  Any failing
        error check fails the run.
    WARNING
        A signal worth reading before trusting the feed, but one a sound
        capture can legitimately show (orders resting before the capture
        began, zero-priced levels, messages reordered in transit).  Fails the
        run only under ``ob-analytics audit --strict``.
    INFO
        Reported for context; never fails a run.
    """

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True)
class QualityCheck:
    """One named data-quality check and how it read on this run.

    Attributes
    ----------
    name : str
        Stable identifier, matching the summary field it reads
        (e.g. ``"duplicate_event_ids"``).
    passed : bool
        Whether the data satisfied the check.  ``False`` for a skipped check,
        which did not pass.
    severity : Severity
        What a failure means (see :class:`Severity`).
    detail : str
        One line saying what was found and how to read it.
    skipped : bool
        Whether the check did not run, because the data cannot answer it.
        A skipped check is :attr:`Severity.INFO` and never fails a run;
        *detail* says why it did not run.
    """

    name: str
    passed: bool
    severity: Severity
    detail: str
    skipped: bool = False

    @classmethod
    def skip(cls, name: str, why: str) -> QualityCheck:
        """A check that did not run, and *why*."""
        return cls(name, False, Severity.INFO, why, skipped=True)

    def to_dict(self) -> dict[str, Any]:
        """Return the check as a plain, JSON-serialisable dict."""
        return {
            "name": self.name,
            "passed": self.passed,
            "skipped": self.skipped,
            "severity": str(self.severity.value),
            "detail": self.detail,
        }
