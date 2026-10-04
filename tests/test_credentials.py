"""API keys: how a source declares them, and that no capture file holds one (#239)."""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import pytest
from loguru import logger
from pydantic import SecretStr

from ob_analytics import ConfigError, Credential, SourceSettings
from ob_analytics._secrets import MASK, redact, remember
from ob_analytics.live import CaptureConfig, run_capture
from ob_analytics.live._base import EventDict
from ob_analytics.live._runner import run_capturer
from ob_analytics.protocols import FeedType, Level

KEY_ENV = "OBA_TEST_API_KEY"
KEY_FILE_ENV = "OBA_TEST_PRIVATE_KEY_PATH"
ISSUED_AT = "https://venue.example/account/api-keys"

FAKE_KEY = "fake-key-7f3a9c2e51d84b06"
FAKE_PEM = (
    "-----BEGIN PRIVATE KEY-----\n"
    "FAKEPEMBODYq8Zt3XyQ1vN0mLk7Hs2Rj5Wd\n"
    "-----END PRIVATE KEY-----\n"
)

REPO_ROOT = Path(__file__).resolve().parents[1]


class _KeyedSettings(SourceSettings):
    api_key_id: Annotated[
        SecretStr | None, Credential(env=KEY_ENV, issued_at=ISSUED_AT)
    ] = None
    private_key: Annotated[
        SecretStr | None,
        Credential(env=KEY_FILE_ENV, issued_at=ISSUED_AT, from_file=True),
    ] = None
    depth: int = 10


class _LeakySource:
    """An L3 source that puts its own key everywhere a careless one could.

    The key goes into a raw frame, a trade id, a log line, its diagnostics and
    the error its close-out raises.  None of it may reach a file.
    """

    name = "keyed"
    level = Level.L3
    feed_type = FeedType.MATCHED_BOOK

    def __init__(self, settings: _KeyedSettings | None = None) -> None:
        self.settings: SourceSettings = settings or _KeyedSettings()

    def _secrets(self) -> tuple[str, str]:
        assert isinstance(self.settings, _KeyedSettings)
        key = self.settings.api_key_id
        pem = self.settings.private_key
        assert key is not None and pem is not None
        return key.get_secret_value(), pem.get_secret_value()

    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        ts = pd.Timestamp.now(tz="UTC")
        yield {
            "id": 1,
            "timestamp": ts,
            "exchange_timestamp": ts,
            "price": 100.0,
            "volume": 1.0,
            "action": "created",
            "direction": "bid",
        }

    async def stream(
        self, config: CaptureConfig
    ) -> AsyncIterator[tuple[str, EventDict, Any]]:
        key, pem = self._secrets()
        logger.warning("venue accepted key {} for {!r}", key, self.settings)
        ts = pd.Timestamp.now(tz="UTC")
        auth_echo = {"type": "subscribed", "key": key, "signed_with": pem}
        yield (
            "order",
            {
                "id": 2,
                "timestamp": ts,
                "exchange_timestamp": ts,
                "price": 101.0,
                "volume": 1.0,
                "action": "created",
                "direction": "ask",
            },
            auth_echo,
        )
        yield (
            "trade",
            {
                "trade_id": f"t-{key}",
                "timestamp": ts,
                "exchange_timestamp": ts,
                "price": 100.0,
                "amount": 0.5,
                "buy_order_id": 1,
                "sell_order_id": 2,
                "side": "sell",
            },
            None,
        )
        await asyncio.sleep(3600)  # until the capture stops

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        key, _ = self._secrets()
        raise ConnectionError(f"logout refused for key {key}")
        yield {}  # pragma: no cover - makes this an async generator

    def diagnostics(self) -> dict[str, Any]:
        key, _ = self._secrets()
        return {"session": f"session-for-{key}"}


class _SettingsFreeSource(_LeakySource):
    """Builds its own settings, as the protocol allows: it takes no arguments."""

    name = "keyed-no-arg"

    def __init__(self) -> None:
        super().__init__()

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        return
        yield {}  # pragma: no cover - makes this an async generator


class _KeyFileRemovingSource(_LeakySource):
    """Removes its key file once it has read it, as a key rotation would."""

    name = "keyed-rotating"

    async def snapshot(self, config: CaptureConfig) -> AsyncIterator[EventDict]:
        Path(os.environ[KEY_FILE_ENV]).unlink(missing_ok=True)
        async for event in super().snapshot(config):
            yield event

    async def shutdown_synthetic_events(self) -> AsyncIterator[EventDict]:
        return
        yield {}  # pragma: no cover - makes this an async generator


@pytest.fixture
def no_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(KEY_ENV, raising=False)
    monkeypatch.delenv(KEY_FILE_ENV, raising=False)


@pytest.fixture
def key_file(tmp_path: Path) -> Path:
    path = tmp_path / "venue.pem"
    path.write_text(FAKE_PEM)
    return path


# ---------------------------------------------------------------------------
# Declaring and reading credentials
# ---------------------------------------------------------------------------


class TestDeclaration:
    def test_credentials_lists_the_marked_fields(self) -> None:
        creds = _KeyedSettings.credentials()
        assert list(creds) == ["api_key_id", "private_key"]
        assert creds["api_key_id"].env == KEY_ENV
        assert creds["private_key"].from_file

    def test_settings_without_credentials_declare_none(self) -> None:
        assert SourceSettings.credentials() == {}

    def test_a_credential_must_be_a_secret(self) -> None:
        with pytest.raises(TypeError, match=re.escape("'SecretStr | None'")):

            class _Plain(SourceSettings):
                key: Annotated[str | None, Credential(env="X", issued_at=ISSUED_AT)] = (
                    None
                )

    def test_a_credential_must_default_to_none(self) -> None:
        with pytest.raises(TypeError, match="default of None"):

            class _Required(SourceSettings):
                key: Annotated[
                    SecretStr | None, Credential(env="X", issued_at=ISSUED_AT)
                ]


class TestReading:
    def test_unset_is_none(self, no_key_env: None) -> None:
        settings = _KeyedSettings()
        assert settings.api_key_id is None
        assert settings.private_key is None

    def test_read_from_the_environment(
        self, monkeypatch: pytest.MonkeyPatch, key_file: Path
    ) -> None:
        monkeypatch.setenv(KEY_ENV, FAKE_KEY)
        monkeypatch.setenv(KEY_FILE_ENV, str(key_file))
        settings = _KeyedSettings()
        assert settings.api_key_id == SecretStr(FAKE_KEY)
        assert settings.private_key == SecretStr(FAKE_PEM)

    def test_an_empty_variable_is_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(KEY_ENV, "")
        monkeypatch.setenv(KEY_FILE_ENV, "")
        assert _KeyedSettings().api_key_id is None
        assert _KeyedSettings().private_key is None

    def test_a_value_given_wins_over_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(KEY_ENV, FAKE_KEY)
        settings = _KeyedSettings(api_key_id=SecretStr("given-key-0123456789"))
        assert settings.api_key_id == SecretStr("given-key-0123456789")

    def test_a_path_given_for_a_key_file_is_read(
        self, no_key_env: None, key_file: Path
    ) -> None:
        settings = _KeyedSettings(private_key=key_file)
        assert settings.private_key == SecretStr(FAKE_PEM)

    def test_an_empty_value_given_reads_the_environment(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(KEY_ENV, FAKE_KEY)
        assert _KeyedSettings(api_key_id="").api_key_id == SecretStr(FAKE_KEY)
        assert _KeyedSettings(api_key_id=SecretStr("")).api_key_id == SecretStr(
            FAKE_KEY
        )

    def test_a_key_too_short_to_be_real_is_refused(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv(KEY_ENV, "test")
        with pytest.raises(ConfigError, match=KEY_ENV):
            _KeyedSettings()

    def test_a_key_file_that_is_missing_names_the_variable_and_path(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        missing = tmp_path / "nowhere.pem"
        monkeypatch.setenv(KEY_FILE_ENV, str(missing))
        with pytest.raises(ConfigError, match=KEY_FILE_ENV) as err:
            _KeyedSettings()
        assert str(missing) in str(err.value)

    def test_an_empty_key_file_is_refused(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty.pem"
        empty.write_text("\n")
        monkeypatch.setenv(KEY_FILE_ENV, str(empty))
        with pytest.raises(ConfigError, match="empty"):
            _KeyedSettings()

    def test_the_key_does_not_print(
        self, monkeypatch: pytest.MonkeyPatch, key_file: Path
    ) -> None:
        monkeypatch.setenv(KEY_ENV, FAKE_KEY)
        monkeypatch.setenv(KEY_FILE_ENV, str(key_file))
        settings = _KeyedSettings()
        for text in (
            repr(settings),
            str(settings),
            json.dumps(settings.model_dump(mode="json")),
            settings.model_dump_json(),
        ):
            assert FAKE_KEY not in text
            assert "FAKEPEMBODY" not in text


class TestMissing:
    def test_check_names_each_variable_and_where_keys_are_issued(
        self, no_key_env: None
    ) -> None:
        with pytest.raises(ConfigError) as err:
            _KeyedSettings().check_credentials("keyed")
        message = str(err.value)
        assert "'keyed'" in message
        assert KEY_ENV in message
        assert KEY_FILE_ENV in message
        assert ISSUED_AT in message

    def test_check_passes_when_every_key_is_set(
        self, monkeypatch: pytest.MonkeyPatch, key_file: Path
    ) -> None:
        monkeypatch.setenv(KEY_ENV, FAKE_KEY)
        monkeypatch.setenv(KEY_FILE_ENV, str(key_file))
        _KeyedSettings().check_credentials("keyed")

    def test_run_capture_stops_before_any_output(
        self, no_key_env: None, tmp_path: Path
    ) -> None:
        out = tmp_path / "cap"
        config = CaptureConfig(pair="x", out_dir=out, minutes=0.01)
        with pytest.raises(ConfigError, match=KEY_ENV):
            asyncio.run(run_capture(_LeakySource, config))
        assert not out.exists()

    def test_run_capturer_stops_before_any_output(
        self, no_key_env: None, tmp_path: Path
    ) -> None:
        out = tmp_path / "seg"
        config = CaptureConfig(pair="x", out_dir=out, minutes=0.01)
        with pytest.raises(ConfigError, match=KEY_ENV):
            asyncio.run(run_capturer(_LeakySource(), config))
        assert not out.exists()


class TestRedact:
    def test_a_short_value_is_never_redacted(self) -> None:
        remember("1234567")
        assert redact("price 1234567") == "price 1234567"

    def test_a_key_set_without_validation_is_redacted_once_checked(
        self, no_key_env: None, key_file: Path
    ) -> None:
        copied_key = "copied-key-5be1d07c93aa4f12"
        settings = _KeyedSettings(private_key=key_file).model_copy(
            update={"api_key_id": SecretStr(copied_key)}
        )
        settings.check_credentials("keyed")
        assert redact(copied_key) == MASK

    def test_a_key_with_a_newline_is_found_json_escaped(self) -> None:
        _KeyedSettings(api_key_id=SecretStr(FAKE_KEY), private_key=SecretStr(FAKE_PEM))
        text = json.dumps({"key": FAKE_KEY, "pem": FAKE_PEM})
        assert redact(text) == json.dumps({"key": MASK, "pem": MASK})


# ---------------------------------------------------------------------------
# The whole capture, through the CLI
# ---------------------------------------------------------------------------

# Registers the leaky source, then runs the CLI as `ob-analytics` would.
_RUN_CLI = (
    "import sys; "
    "from ob_analytics.sources import register_source; "
    "import tests.test_credentials as t; "
    "from tests.test_credentials import _LeakySource; "
    "register_source('keyed', _LeakySource); "
    "register_source('keyed-rotating', t._KeyFileRemovingSource); "
    "register_source('keyed-no-arg', t._SettingsFreeSource); "
    "from ob_analytics.cli import main; "
    "sys.argv[0] = 'ob-analytics'; "
    "main()"
)


def _cli(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", _RUN_CLI, *args],
        capture_output=True,
        text=True,
        check=False,
        cwd=REPO_ROOT,
        env=env,
        timeout=120,
    )


def _clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in (KEY_ENV, KEY_FILE_ENV)}
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(REPO_ROOT), env.get("PYTHONPATH")) if p
    )
    return env


class TestCaptureWritesNoKey:
    def test_no_file_and_no_log_line_holds_the_key(
        self, tmp_path: Path, key_file: Path
    ) -> None:
        out = tmp_path / "cap"
        env = {**_clean_env(), KEY_ENV: FAKE_KEY, KEY_FILE_ENV: str(key_file)}
        proc = _cli(
            "capture", "keyed", "--pair", "x", "--out", str(out),
            "--minutes", "0.03",
            env=env,
        )  # fmt: skip
        assert proc.returncode == 0, proc.stderr

        files = [p for p in out.rglob("*") if p.is_file()]
        names = {p.name for p in files}
        assert {"orders.csv", "trades.csv", "raw.jsonl", "meta.json"} <= names
        assert "manifest.json" in names

        # Every form the key could take in a file: as is, and JSON-escaped.
        forbidden = {FAKE_KEY, FAKE_PEM, json.dumps(FAKE_PEM)[1:-1], "FAKEPEMBODY"}
        for path in files:
            text = path.read_bytes().decode("utf-8", errors="replace")
            for secret in forbidden:
                assert secret not in text, f"{secret!r} found in {path}"
        for secret in forbidden:
            assert secret not in proc.stderr, f"{secret!r} found in the log"

        # The leaks did happen, and were masked rather than dropped.
        seg = out / "seg-0001"
        assert MASK in (seg / "raw.jsonl").read_text()
        assert MASK in (seg / "trades.csv").read_text()
        meta = json.loads((seg / "meta.json").read_text())
        assert meta["session"] == f"session-for-{MASK}"
        assert meta["capture_error"] == repr(
            ConnectionError(f"logout refused for key {MASK}")
        )
        manifest = json.loads((out / "manifest.json").read_text())
        assert MASK in manifest["segments"][0]["error"]
        assert f"venue accepted key {MASK}" in proc.stderr

    def test_a_missing_key_stops_the_cli_before_any_output(
        self, tmp_path: Path
    ) -> None:
        out = tmp_path / "cap"
        proc = _cli(
            "capture", "keyed", "--pair", "x", "--out", str(out),
            "--minutes", "0.03",
            env=_clean_env(),
        )  # fmt: skip
        assert proc.returncode == 1
        assert KEY_ENV in proc.stderr
        assert KEY_FILE_ENV in proc.stderr
        assert ISSUED_AT in proc.stderr
        assert not out.exists()

    def test_a_key_file_removed_mid_capture_does_not_end_it(
        self, tmp_path: Path, key_file: Path
    ) -> None:
        out = tmp_path / "cap"
        env = {**_clean_env(), KEY_ENV: FAKE_KEY, KEY_FILE_ENV: str(key_file)}
        proc = _cli(
            "capture", "keyed-rotating", "--pair", "x", "--out", str(out),
            "--minutes", "0.05", "--roll-minutes", "0.01",
            env=env,
        )  # fmt: skip
        assert proc.returncode == 0, proc.stderr
        assert not key_file.exists()
        manifest = json.loads((out / "manifest.json").read_text())
        assert len(manifest["segments"]) >= 2

    def test_a_source_that_takes_no_settings_rolls(
        self, tmp_path: Path, key_file: Path
    ) -> None:
        out = tmp_path / "cap"
        env = {**_clean_env(), KEY_ENV: FAKE_KEY, KEY_FILE_ENV: str(key_file)}
        proc = _cli(
            "capture", "keyed-no-arg", "--pair", "x", "--out", str(out),
            "--minutes", "0.05", "--roll-minutes", "0.01",
            env=env,
        )  # fmt: skip
        assert proc.returncode == 0, proc.stderr
        manifest = json.loads((out / "manifest.json").read_text())
        assert len(manifest["segments"]) >= 2
        assert all(s["error"] is None for s in manifest["segments"]), manifest

    def test_sources_reports_a_key_file_that_cannot_be_read(
        self, tmp_path: Path
    ) -> None:
        env = {**_clean_env(), KEY_FILE_ENV: str(tmp_path / "gone.pem")}
        proc = _cli("sources", env=env)
        assert proc.returncode == 0, proc.stderr
        line = next(s for s in proc.stdout.splitlines() if s.startswith("keyed "))
        assert "cannot be built" in line
        assert KEY_FILE_ENV in line
        assert any(s.startswith("bitstamp") for s in proc.stdout.splitlines())

    def test_sources_lists_the_key_variables(self) -> None:
        proc = _cli("sources", env=_clean_env())
        assert proc.returncode == 0, proc.stderr
        line = next(s for s in proc.stdout.splitlines() if s.startswith("keyed "))
        assert f"key: {KEY_ENV}, {KEY_FILE_ENV}" in line
