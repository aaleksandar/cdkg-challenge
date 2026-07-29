"""GitHub App authentication and the PR flow. Implemented in a later step."""

from __future__ import annotations


def publish_ingest(ctx: dict):
    from .pipeline.stages import StageResult

    return StageResult(False, "Publishing not yet implemented")
