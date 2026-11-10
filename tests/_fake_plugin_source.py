"""A live source of the kind a third-party package ships, for the CLI tests.

It is never streamed: the tests replace the capture loop.  What matters is its
settings class, whose fields cover each kind of flag ``capture`` builds.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, SecretStr

from ob_analytics.config import Credential, SourceSettings
from ob_analytics.protocols import FeedType, Level
from ob_analytics.sources import CliOption

#: The environment variable the fake source's key is read from.
TOKEN_ENV = "OB_ANALYTICS_FAKE_PLUGIN_TOKEN"


class FakePluginSettings(SourceSettings):
    market: Annotated[str, CliOption(help="The market to follow")]
    depth: Annotated[
        int, CliOption(help="Levels a side", flag="levels"), Field(gt=0)
    ] = 10
    speed: float = 1.0
    book_side: Literal["bid", "ask"] = "bid"
    resolution: Level | None = None
    fast: bool = False
    where: Path | None = None
    token: Annotated[
        SecretStr | None, Credential(env=TOKEN_ENV, issued_at="https://example.test")
    ] = None
    handle: Any = None


class FakePluginSource:
    name = "fakeplugin"
    level = Level.L3
    feed_type = FeedType.MATCHED_BOOK

    # The settings have a field with no default, so the source cannot build
    # them for itself: it takes them, and names their class for the command
    # line to read.
    def __init__(self, settings: FakePluginSettings) -> None:
        self.settings: SourceSettings = settings

    async def snapshot(self, config: Any) -> AsyncIterator[dict[str, Any]]:
        return
        yield {}  # pragma: no cover - makes this an async generator

    async def stream(self, config: Any) -> AsyncIterator[tuple[str, Any, Any]]:
        await asyncio.sleep(3600)
        yield ("order", {}, None)  # pragma: no cover

    async def shutdown_synthetic_events(self) -> AsyncIterator[dict[str, Any]]:
        return
        yield {}  # pragma: no cover


class ClashingSettings(SourceSettings):
    minutes: int = 3


class ClashingSource(FakePluginSource):
    """Names no settings class, so the command line reads a default's."""

    name = "clashing"

    def __init__(self, settings: SourceSettings | None = None) -> None:
        self.settings = settings or ClashingSettings()


class OptionalSettings(SourceSettings):
    depth: int = 5


class SettingsRequiredSource(FakePluginSource):
    """Takes its settings as a required argument, though every field has a default."""

    name = "settingsrequired"

    def __init__(self, settings: OptionalSettings) -> None:
        self.settings = settings
