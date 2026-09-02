"""What an ingestion cost: tokens from BAML, money from configured rates.

Tokens are a fact the provider reports and BAML hands back through a Collector,
so they are recorded unconditionally and are always exact.

Money is not a fact this codebase can know. Prices change, they differ per model
and per region, and a hardcoded rate silently becomes a lie the moment Google
moves one. So the rates live in the environment and a cost is shown only when
they have been set — a blank cost means "nobody told me the price", which is
honest, where a confidently wrong number is not.

    INGEST_INPUT_COST_PER_MTOK=0.30     # USD per million input tokens
    INGEST_OUTPUT_COST_PER_MTOK=2.50    # USD per million output tokens
    INGEST_CACHED_INPUT_COST_PER_MTOK=0.075

Find the current numbers on the provider's pricing page for the model pinned in
clients.baml, which `ingest.model.tag_model()` reports.
"""

from __future__ import annotations

from . import config


def _rate(raw: str | None) -> float | None:
    try:
        return float(raw) if raw not in (None, "") else None
    except ValueError:
        return None


def rates_configured() -> bool:
    return (_rate(config.INPUT_COST_PER_MTOK) is not None
            or _rate(config.OUTPUT_COST_PER_MTOK) is not None)


def usage_of(collector) -> dict:
    """Token counts from a BAML Collector, as a plain dict for the run record.

    Never raises: a missing usage figure must not fail an ingestion that has
    already done its work and been paid for.
    """
    try:
        usage = collector.usage
    except Exception:  # noqa: BLE001
        return {}

    tokens = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "cached_input_tokens": getattr(usage, "cached_input_tokens", None),
    }
    tokens = {k: v for k, v in tokens.items() if v is not None}
    if not tokens:
        return {}

    cost = estimate_cost(tokens)
    if cost is not None:
        tokens["cost_usd"] = cost
    return tokens


def estimate_cost(tokens: dict) -> float | None:
    """USD for these tokens, or None when the rates have not been configured.

    Cached input is billed at its own lower rate when one is given, and is
    subtracted from the plain input count so it is never charged twice.
    """
    inp = _rate(config.INPUT_COST_PER_MTOK)
    out = _rate(config.OUTPUT_COST_PER_MTOK)
    cached_rate = _rate(config.CACHED_INPUT_COST_PER_MTOK)
    if inp is None and out is None:
        return None

    cached = tokens.get("cached_input_tokens") or 0
    fresh_input = max((tokens.get("input_tokens") or 0) - cached, 0)

    total = 0.0
    total += fresh_input / 1_000_000 * (inp or 0.0)
    total += (tokens.get("output_tokens") or 0) / 1_000_000 * (out or 0.0)
    total += cached / 1_000_000 * (cached_rate if cached_rate is not None else (inp or 0.0))
    return round(total, 6)


def format_cost(amount: float | None) -> str:
    """Small amounts read as fractions of a cent, not as $0.00."""
    if amount is None:
        return "—"
    if amount < 0.01:
        return f"{amount * 100:.2f}¢"
    return f"${amount:.2f}"
