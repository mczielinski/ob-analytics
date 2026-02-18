"""Custom exception hierarchy for the ob-analytics package.

All exceptions inherit from :class:`ObAnalyticsError`, allowing callers to
catch every package-specific error with a single ``except`` clause.
"""


class ObAnalyticsError(Exception):
    """Base exception for all ob-analytics errors."""


class InvalidDataError(ObAnalyticsError):
    """Input data is missing required columns or has invalid values."""


class MatchingError(ObAnalyticsError):
    """Event matching or trade inference encountered an unrecoverable problem."""


class InsufficientDataError(ObAnalyticsError):
    """Not enough data to perform the requested operation."""


class ConfigurationError(ObAnalyticsError):
    """Pipeline configuration values are invalid or inconsistent."""
