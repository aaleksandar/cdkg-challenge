"""What an ingestion cost.

Tokens are a fact the provider reports. Money is not something this codebase can
know, so these tests are mostly about it refusing to invent one.
"""

import pytest

from ingest import config, spend


@pytest.fixture
def priced(monkeypatch):
    """Rates as an operator would configure them, per million tokens."""
    monkeypatch.setattr(config, "INPUT_COST_PER_MTOK", "0.75")
    monkeypatch.setattr(config, "OUTPUT_COST_PER_MTOK", "3.75")
    monkeypatch.setattr(config, "CACHED_INPUT_COST_PER_MTOK", "0.075")


@pytest.fixture
def unpriced(monkeypatch):
    for name in ("INPUT_COST_PER_MTOK", "OUTPUT_COST_PER_MTOK",
                 "CACHED_INPUT_COST_PER_MTOK"):
        monkeypatch.setattr(config, name, None)


def test_without_rates_there_is_no_cost_rather_than_a_wrong_one(unpriced):
    """A hardcoded price becomes a lie the moment the provider moves it. Blank
    means "nobody told me the price", which is honest."""
    assert spend.estimate_cost({"input_tokens": 1060, "output_tokens": 47}) is None
    assert spend.rates_configured() is False
    assert spend.format_cost(None) == "—"


def test_cost_is_computed_per_million_tokens(priced):
    cost = spend.estimate_cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    assert cost == pytest.approx(0.75 + 3.75)


def test_cached_input_is_billed_at_its_own_rate_and_only_once(priced):
    """Providers report cached tokens inside the input count. Charging both the
    full rate and the cached rate would overstate every large transcript."""
    cost = spend.estimate_cost({
        "input_tokens": 10_000, "output_tokens": 0, "cached_input_tokens": 10_000,
    })
    assert cost == pytest.approx(10_000 / 1_000_000 * 0.075)

    partial = spend.estimate_cost({
        "input_tokens": 10_000, "output_tokens": 0, "cached_input_tokens": 4_000,
    })
    assert partial == pytest.approx(6_000 / 1e6 * 0.75 + 4_000 / 1e6 * 0.075)


def test_a_fraction_of_a_cent_reads_as_a_fraction_of_a_cent(priced):
    """Every one of these calls rounds to $0.00, which tells an admin nothing."""
    assert spend.format_cost(0.000435) == "0.04¢"
    assert spend.format_cost(0.0056) == "0.56¢"
    assert spend.format_cost(1.5) == "$1.50"


def test_a_collector_without_usage_does_not_break_a_finished_run(unpriced):
    """The work is done and already paid for by the time this is read."""
    class Broken:
        @property
        def usage(self):
            raise RuntimeError("no usage reported")

    assert spend.usage_of(Broken()) == {}

    class Empty:
        usage = type("U", (), {"input_tokens": None, "output_tokens": None,
                               "cached_input_tokens": None})()

    assert spend.usage_of(Empty()) == {}


def test_usage_carries_the_cost_only_when_it_can(priced, unpriced):
    """`unpriced` is applied last, so rates are unset here."""
    class C:
        usage = type("U", (), {"input_tokens": 100, "output_tokens": 10,
                               "cached_input_tokens": None})()

    got = spend.usage_of(C())
    assert got == {"input_tokens": 100, "output_tokens": 10}
    assert "cost_usd" not in got
