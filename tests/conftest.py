"""Guards shared by every test: nothing a test does may land in the real tree."""

import pytest

from ingest import config


@pytest.fixture(autouse=True)
def _snapshots_in_tmp(monkeypatch, tmp_path):
    """Snapshots default to a folder beside the real database; tests get their own."""
    monkeypatch.setattr(config, "SNAPSHOT_DIR", tmp_path / "snapshots")


@pytest.fixture(autouse=True)
def _panel_open_for_tests(monkeypatch):
    """Tests drive the panel anonymously, whatever a local .env says: an
    ADMIN_PASSWORD there would otherwise turn every panel test into a 401."""
    monkeypatch.setattr(config, "ADMIN_PASSWORD", None)
    monkeypatch.setattr(config, "ALLOW_ANONYMOUS_PANEL", True)


@pytest.fixture(autouse=True)
def _no_network_no_keys(monkeypatch):
    """A test that forgets a stub fails loudly instead of spending. The real
    keys come from src/kuzu/.env, which config loads at import; a call that
    slipped through used to reach Gemini, Supadata or HeySummit with them."""
    import socket

    for name in ("GOOGLE_API_KEY", "SUPADATA_API_KEY", "HEYSUMMIT_API_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config, "SUPADATA_API_KEY", None)
    monkeypatch.setattr(config, "HEYSUMMIT_API_TOKEN", None)

    real_connect = socket.socket.connect

    def local_only(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if self.family == socket.AF_UNIX or host in ("127.0.0.1", "::1", "localhost"):
            return real_connect(self, address)
        raise RuntimeError(f"Test tried to reach the network: {address}")
    monkeypatch.setattr(socket.socket, "connect", local_only)
