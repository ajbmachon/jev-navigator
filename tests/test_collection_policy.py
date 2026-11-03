"""Real Umami schema proves bounded composition, admission and duplicate backfill."""

from pathlib import Path

import pytest

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.index.units import Reading, UnitReader, read_ranges
from jev_navigator.selection import RankFeatures, RankWeights, ScentIndex, scent_document
from jev_navigator.selection.collection import collect_diverse
from jev_navigator.sources import Seeds


@pytest.fixture
def schema():
    root = Path(__file__).parent / "fixtures/umami"
    index = CodeIndex(root, ["schema.prisma"])
    reader = UnitReader(index, 76800, listed_only=True, reading=Reading.MIXED)
    units = {u.id: u for u in reader.list_files(["schema.prisma"]).units}
    scent = ScentIndex(
        scent_document(u.id, u.path, u.symbol, read_ranges(index, u.path, u.ranges)) for u in units.values()
    )
    scores = {"model ApiKey": 3, "model Link": 2, "model Session": 1}
    features = {key: RankFeatures(prior=scores.get(u.symbol, 0)) for key, u in units.items()}
    return index, reader, units, scent, features


def test_known_word_and_file_routes_preserve_rank_prefix(schema):
    index, reader, units, scent, features = schema
    result = collect_diverse(
        index,
        reader,
        Seeds(files=("schema.prisma",)),
        scent,
        list(units),
        units,
        features,
        RankWeights(scent=0, walk=0, path=0, prior=1),
        text="distinctId",
        cited_text="",
        limit=3,
        reserve=2,
    )
    assert [units[key].symbol for key in result.ids] == ["model ApiKey", "model Session", "model Link"]
    # The schema's whole-model bodies, rather than just declaration locations, remain readable.
    session = units[result.ids[1]]
    assert "distinctId" in read_ranges(index, session.path, session.ranges)


def test_no_route_can_admit_an_excluded_model_and_empty_routes_backfill(schema):
    index, reader, units, scent, features = schema
    admitted = [
        key for key, u in units.items() if u.symbol in {"model ApiKey", "model Link", "model Session"}
    ]
    result = collect_diverse(
        index,
        reader,
        Seeds(names=("Website",)),
        scent,
        admitted,
        units,
        features,
        RankWeights(scent=0, walk=0, path=0, prior=1),
        text="",
        cited_text="",
        limit=3,
        reserve=2,
    )
    assert set(result.ids) == set(admitted)
    assert len(result.ids) == len(set(result.ids)) == 3
    assert result.ids[0] == next(key for key in admitted if units[key].symbol == "model ApiKey")
