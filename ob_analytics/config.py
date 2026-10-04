"""Pipeline configuration for ob-analytics.

Centralises the numeric thresholds and parameters that were previously
scattered as literals across multiple modules.
"""

import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Self, Union, get_args, get_origin

from pydantic import BaseModel, Field, SecretStr, model_validator

from ob_analytics._secrets import MIN_SECRET_LENGTH, remember
from ob_analytics.exceptions import ConfigError


@dataclass(frozen=True)
class Credential:
    """Marks a :class:`SourceSettings` field as a key the user supplies.

    Put it on a ``SecretStr | None`` field with ``typing.Annotated``.  When the
    field is not given, the settings read it from the environment variable
    *env*.  A key is never a command-line option, because a command line stays
    in the shell history::

        class VenueSettings(SourceSettings):
            api_key: Annotated[
                SecretStr | None,
                Credential(env="VENUE_API_KEY", issued_at="https://venue.example/keys"),
            ] = None

    An empty value counts as unset.  A value shorter than eight characters is
    refused, since no venue issues a key that short.  A capture checks that
    every credential is set before it writes anything
    (:meth:`SourceSettings.check_credentials`).

    Attributes
    ----------
    env : str
        The environment variable the key is read from.
    issued_at : str
        Where the venue issues keys, usually a URL.  The error for a missing
        key names it.
    from_file : bool
        The variable holds the path of a file, and the field holds the file's
        contents.  For a private key, which lives in a file.  From Python, pass
        a :class:`~pathlib.Path` to read a file, or a ``str`` that is the key
        itself.
    """

    env: str
    issued_at: str
    from_file: bool = False


class SourceSettings(BaseModel):
    """Base class for a data source's typed, immutable settings.

    The typed replacement for the untyped per-source settings dict that live
    capturers used to carry (``CaptureConfig.extras``).  Every
    :class:`~ob_analytics.protocols.Source` declares a ``settings`` value of
    this type; a source that needs no configuration uses the empty base, and a
    source with venue knobs subclasses it with typed, validated fields — e.g.
    :class:`~ob_analytics.live.ccxt_source.CcxtSettings` (``exchange`` /
    ``depth_limit`` / ``poll_interval``).

    Frozen so a source's settings are fixed for the run, matching
    :class:`PipelineConfig`.

    A source that needs an API key declares it as a field marked with
    :class:`Credential`.  The settings read an unset credential from its
    environment variable, and every credential's value is removed from the
    files a capture writes.
    """

    model_config = {"frozen": True}

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        for name, field in cls.model_fields.items():
            if not any(isinstance(m, Credential) for m in field.metadata):
                continue
            # A key in a plain str would print in a repr or a log line, and a
            # field with a default other than None would fail to build when
            # the key is not set, which is every run that does not capture.
            if not _is_optional_secret(field.annotation) or field.default is not None:
                raise TypeError(
                    f"{cls.__name__}.{name} is a Credential, so it must be typed "
                    "'SecretStr | None' with a default of None"
                )

    @classmethod
    def credentials(cls) -> dict[str, Credential]:
        """The fields marked with :class:`Credential`, by field name."""
        return {
            name: marker
            for name, field in cls.model_fields.items()
            for marker in field.metadata
            if isinstance(marker, Credential)
        }

    @model_validator(mode="before")
    @classmethod
    def _read_credentials(cls, data: Any) -> Any:
        credentials = cls.credentials()
        if not credentials or not isinstance(data, dict):
            return data
        data = dict(data)
        for name, credential in credentials.items():
            value = data.get(name)
            if isinstance(value, SecretStr):
                value = value.get_secret_value()
            if value is None or value == "":
                data[name] = _from_environment(credential)
            elif credential.from_file and isinstance(value, Path):
                data[name] = _read_key_file(value, f"{cls.__name__}.{name}")
        return data

    @model_validator(mode="after")
    def _check_and_remember(self) -> Self:
        for name, credential in type(self).credentials().items():
            value = getattr(self, name)
            if value is not None and len(value.get_secret_value()) < MIN_SECRET_LENGTH:
                raise ConfigError(
                    f"{type(self).__name__}.{name} is shorter than "
                    f"{MIN_SECRET_LENGTH} characters, which is not a venue key; "
                    f"check the value given, or {credential.env} if it came "
                    "from the environment"
                )
        self.remember_secrets()
        return self

    def remember_secrets(self) -> None:
        """Register every ``SecretStr`` value held here for redaction.

        Validation does this already.  A capture does it again as it starts,
        for settings built without validation (``model_copy(update=...)``).
        """
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, SecretStr):
                remember(value.get_secret_value())

    def check_credentials(self, source: str) -> None:
        """Raise :class:`~ob_analytics.exceptions.ConfigError` if a credential is unset.

        The message names each missing key's environment variable and where
        the venue issues keys.  *source* is the source's name, for the message.
        The keys that are set are registered for redaction.
        """
        self.remember_secrets()
        missing = [
            (name, credential)
            for name, credential in type(self).credentials().items()
            if getattr(self, name) is None
        ]
        if not missing:
            return
        lines = [f"Source {source!r} needs a key that is not set:"]
        for name, credential in missing:
            what = (
                f"set {credential.env} to the path of the key file"
                if credential.from_file
                else f"set the environment variable {credential.env}"
            )
            lines.append(f"  {name}: {what}. Get a key at {credential.issued_at}")
        lines.append('See "Use an API key" in the ob-analytics documentation.')
        raise ConfigError("\n".join(lines))


def _is_optional_secret(annotation: Any) -> bool:
    """Whether *annotation* is ``SecretStr | None``."""
    if get_origin(annotation) not in (Union, types.UnionType):
        return False
    return set(get_args(annotation)) == {SecretStr, type(None)}


def _from_environment(credential: Credential) -> str | None:
    """The credential's value from its environment variable, or ``None``."""
    value = os.environ.get(credential.env, "")
    if not value:
        return None
    if credential.from_file:
        return _read_key_file(Path(value), credential.env)
    return value


def _read_key_file(path: Path, named_by: str) -> str:
    """The text of the key file at *path*; *named_by* says where the path came from."""
    path = path.expanduser()
    try:
        text = path.read_text()
    except OSError as exc:
        raise ConfigError(
            f"{named_by} names the key file {path}, which cannot be read: "
            f"{exc.strerror or exc}"
        ) from None
    if not text.strip():
        raise ConfigError(f"{named_by} names the key file {path}, which is empty")
    return text


class PipelineConfig(BaseModel):
    """Validated, immutable configuration for the ob-analytics pipeline.

    Every parameter that was previously a hard-coded literal now lives here
    with a sensible default matching the original R package behaviour (Bitstamp
    BTC/USD, 2015).  Override individual values for different instruments,
    exchanges, or precision requirements.
    """

    model_config = {"frozen": True}

    # ── Price / volume precision ──────────────────────────────────────────
    tick_size: float = Field(
        default=0.01,
        gt=0,
        description=(
            "The instrument's minimum price increment, in the quote currency "
            "(issue #155).  Prices are stored as a whole number of ticks "
            "(``int64``); the quote-currency price is ``ticks * tick_size``.  "
            "0.01 (default) is a cent grid (USD equities, BTC-USD); use the "
            "venue's real tick for small-tick crypto or 0-1 prediction markets. "
            "By default it matches ``price_decimals`` (``10 ** -price_decimals``)."
        ),
    )
    price_decimals: int = Field(
        default=2,
        ge=0,
        le=18,
        description=(
            "Display precision: decimal places to show when a tick price is "
            "rendered back to the quote currency for a plot or CSV.  2 for USD "
            "equities / BTC-USD; 8 for satoshi-denominated pairs; 4-5 for FX.  "
            "The stored price grid is ``tick_size``, not this — see issue #155."
        ),
    )
    lot_size: float = Field(
        default=1e-8,
        gt=0,
        description=(
            "The instrument's minimum size increment, in the base asset "
            "in the base asset.  Sizes are stored as a whole number of lots "
            "(``int64``); the base-asset size is ``lots * lot_size``.  1e-8 "
            "(default) is a satoshi grid, which is the finest any supported "
            "venue quotes; use 1 for whole shares (LOBSTER).  This is the size "
            "counterpart of ``tick_size``, and exists for the same reason: a "
            "float size does not sum back to exactly zero when a price level "
            "empties, so the level lingers and is reported as the best bid or "
            "ask.  By default it matches ``volume_decimals`` "
            "(``10 ** -volume_decimals``)."
        ),
    )
    volume_decimals: int = Field(
        default=8,
        ge=0,
        le=18,
        description=(
            "Display precision: decimal places to show when a lot size is "
            "rendered back to the base asset for a plot or CSV.  The stored "
            "size grid is ``lot_size``, not this."
        ),
    )
    timestamp_unit: Literal["ms", "us", "ns"] = Field(
        default="ms",
        description=(
            "Unit of raw integer timestamps in the source data.  "
            "'ms' (milliseconds, default) matches Bitstamp CSV format; "
            "'us' for microseconds; 'ns' for nanosecond-precision feeds."
        ),
    )

    price_divisor: int = Field(
        default=1,
        ge=1,
        description=(
            "Raw-feed encoding scale: the divisor that turns a source's raw "
            "integer price into the quote currency, before it is converted to "
            "ticks.  1 (default) means the raw price is already in the quote "
            "currency (Bitstamp).  LOBSTER uses 10 000 (prices are in "
            "ten-thousandths of a dollar).  This is the feed's encoding, "
            "separate from the instrument's ``tick_size``."
        ),
    )

    # ── Sequence / ordering keys ──────────────────────────────────────────
    track_sequence: bool = Field(
        default=False,
        description=(
            "Attach the ordering-key columns to loaded frames (see "
            "ob_analytics.schemas): the local monotonic 'ingest_seq' counter, "
            "and the venue 'sequence' number when the source carries one.  Off "
            "by default so the standard pipeline output is unchanged; turn it "
            "on to detect dropped or reordered messages via "
            "ob_analytics.analytics.detect_sequence_gaps."
        ),
    )

    # ── Depth metrics ─────────────────────────────────────────────────────
    depth_bps: int = Field(
        default=25,
        gt=0,
        description="Width of each depth bin in basis points.",
    )
    depth_bins: int = Field(
        default=20,
        gt=0,
        description="Number of depth bins on each side of the book.",
    )

    # ── Derived helpers ───────────────────────────────────────────────────
    @property
    def price_multiplier(self) -> int:
        """Integer inverse of :attr:`tick_size` (``round(1 / tick_size)``).

        The multiplier that turns a quote-currency price into an integer tick
        count, defined only when the tick is a reciprocal integer (the usual
        case: a cent, nickel, or quarter grid).  With the default ``tick_size``
        of ``0.01`` this is ``100``, matching the former ``10 ** price_decimals``.

        Raises
        ------
        ValueError
            If :attr:`tick_size` has no integer inverse; convert prices with
            ``ob_analytics._utils.price_to_ticks`` (which divides) instead.
        """
        from ob_analytics._utils import tick_multiplier

        multiplier = tick_multiplier(self.tick_size)
        if multiplier is None:
            raise ValueError(
                f"tick_size={self.tick_size!r} has no integer inverse; use "
                "ob_analytics._utils.price_to_ticks to convert prices to ticks."
            )
        return multiplier

    @property
    def bps_labels(self) -> list[str]:
        """Column suffixes for depth-metric BPS bins (e.g. '25bps', '50bps' …)."""
        return [f"{i * self.depth_bps}bps" for i in range(1, self.depth_bins + 1)]
