"""Custom exceptions for the ob-analytics package.

Two exceptions only:

* :class:`ObAnalyticsError` — base for every package-specific error, so
  callers can catch them all with a single ``except`` clause.
* :class:`ConfigError` — invalid or inconsistent configuration *or* input
  data (bad config values, DataFrames missing required columns).
"""


class ObAnalyticsError(Exception):
    """Base exception for all ob-analytics errors."""


class ConfigError(ObAnalyticsError):
    """Configuration values or input data are invalid or inconsistent."""
