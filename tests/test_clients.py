"""Every BAML client names the same model: the live and chat clients differ in
timeouts and thinking, never in the model they ask."""

import re

from ingest import config
from ingest.model import tag_model


def test_every_client_moves_with_the_pinned_model():
    clients = (config.KUZU_DIR / "baml_src" / "clients.baml").read_text(encoding="utf-8")
    models = dict(re.findall(r"client<llm>\s+(\w+)\s*\{.*?\bmodel\s+\"([^\"]+)\"", clients, re.S))

    assert set(models) == {"GeminiFlash", "GeminiFlashLive", "GeminiFlashChat"}
    assert set(models.values()) == {tag_model()}
