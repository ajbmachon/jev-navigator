from __future__ import annotations

from pathlib import Path

import pytest

from jev_navigator.index.file_shape import (
    DENSE_AVERAGE_LINE_CHARS,
    DENSE_MINIMUM_BYTES,
    LARGE_FILE_BYTES,
    LONG_LINE_CHARS,
    MAX_PARSE_PEAK_MB,
    PARSEABLE_UP_TO_BYTES,
    Trigger,
    measure,
    refusal_of,
    shape_of,
)


def _one_line(characters: int) -> bytes:
    statement = b"function f(){return 1};"
    return (statement * (characters // len(statement) + 1))[:characters]


def test_a_file_reports_its_size_line_count_longest_line_and_characters_per_line() -> None:
    shape = measure(b"ab\ncdef\n\ng\n")

    assert (shape.size_bytes, shape.line_count, shape.longest_line) == (11, 4, 4)
    assert shape.chars_per_line == pytest.approx(11 / 4)


def test_an_empty_file_has_no_lines_and_costs_only_the_base() -> None:
    shape = measure(b"")

    assert (shape.line_count, shape.longest_line, shape.chars_per_line) == (0, 0, 0.0)
    assert not shape.too_large_to_parse


@pytest.mark.parametrize(
    ("characters", "measured_mb"),
    [(24_745, 54), (46_329, 122), (88_165, 399), (119_687, 681), (134_721, 956)],
)
def test_the_parse_peak_estimate_fits_the_measured_one_line_bundles_within_ten_percent(
    characters: int, measured_mb: int
) -> None:
    estimate = measure(_one_line(characters)).parse_peak_mb

    assert estimate == pytest.approx(measured_mb, rel=0.10)


def test_many_short_lines_stay_cheap_however_large_the_file_is() -> None:
    shape = measure(b"x = 1\n" * 120_000)

    assert shape.size_bytes == 720_000
    assert shape.parse_peak_mb < 30
    assert not shape.too_large_to_parse


def test_the_bound_sits_between_a_20000_and_a_70000_character_single_line() -> None:
    assert not measure(_one_line(20_000)).too_large_to_parse
    assert not measure(_one_line(65_000)).too_large_to_parse
    assert measure(_one_line(75_000)).too_large_to_parse
    assert measure(_one_line(75_000)).parse_peak_mb > MAX_PARSE_PEAK_MB


def test_the_refusal_names_the_estimated_peak_and_the_longest_line() -> None:
    reason = measure(_one_line(668_777)).refusal

    assert reason == "too large to parse: estimated parse peak 22 GB, longest line 668,777 bytes"
    assert measure(_one_line(20_000)).refusal is None


def _lines(count: int, width: int) -> bytes:
    return b"\n".join(b"x" * width for _ in range(count)) + b"\n"


def test_a_line_over_the_long_line_limit_fires_only_that_trigger_on_a_small_file() -> None:
    content = b"short\n" * 10 + b"y" * (LONG_LINE_CHARS + 1) + b"\n" + b"short\n" * 5000

    assert measure(content).triggers == (Trigger.LONG_LINE,)
    assert measure(b"short\n" * 10 + b"y" * LONG_LINE_CHARS + b"\n" + b"short\n" * 5000).triggers == ()


def test_dense_lines_fire_only_above_the_average_and_the_minimum_size() -> None:
    dense = _lines(count=45, width=DENSE_AVERAGE_LINE_CHARS + 5)
    at_the_average = _lines(count=40, width=DENSE_AVERAGE_LINE_CHARS - 1)
    too_small = _lines(count=30, width=DENSE_AVERAGE_LINE_CHARS + 5)

    assert measure(dense).triggers == (Trigger.DENSE_LINES,)
    assert measure(at_the_average).triggers == ()
    assert measure(too_small).size_bytes < DENSE_MINIMUM_BYTES
    assert measure(too_small).triggers == ()


def test_a_file_over_the_large_file_limit_fires_only_that_trigger_when_its_lines_are_short() -> None:
    over = _lines(count=6_000, width=LARGE_FILE_BYTES // 6_000)
    under = b"x" * 80 + b"\n"
    under = under * ((LARGE_FILE_BYTES // len(under)) - 1)

    assert measure(over).size_bytes > LARGE_FILE_BYTES
    assert measure(over).triggers == (Trigger.LARGE_FILE,)
    assert measure(under).size_bytes <= LARGE_FILE_BYTES
    assert measure(under).triggers == ()


def test_a_hand_written_module_of_many_short_lines_fires_nothing() -> None:
    module = _lines(count=6_000, width=38)

    assert measure(module).size_bytes > 200_000
    assert measure(module).triggers == ()


def test_a_component_with_one_long_svg_path_line_fires_nothing() -> None:
    component = _lines(count=95, width=30) + b"<path d='" + b"M1 2 " * 1_050 + b"'/>\n"

    shape = measure(component)

    assert shape.longest_line > 5_000
    assert shape.size_bytes < LARGE_FILE_BYTES
    assert shape.triggers == ()
    assert not shape.too_large_to_parse


def test_a_one_line_minified_bundle_fires_the_line_and_density_triggers_but_is_still_parseable() -> None:
    shape = measure(_one_line(24_745))

    assert shape.triggers == (Trigger.LONG_LINE, Trigger.DENSE_LINES)
    assert not shape.too_large_to_parse


def test_a_two_and_a_half_megabyte_image_string_fires_every_trigger_and_is_over_the_memory_bound() -> None:
    background = b"export const background = '" + b"A" * 2_504_000 + b"';\n"

    shape = measure(background)

    assert shape.triggers == (Trigger.LONG_LINE, Trigger.DENSE_LINES, Trigger.LARGE_FILE)
    assert shape.too_large_to_parse


def test_shape_of_reads_one_file_given_the_repository_folder_and_the_path(tmp_path: Path) -> None:
    (tmp_path / "dist").mkdir()
    (tmp_path / "dist" / "bundle.js").write_bytes(_one_line(24_745))

    shape = shape_of(tmp_path, "dist/bundle.js")

    assert shape.size_bytes == 24_745
    assert (shape.line_count, shape.longest_line) == (1, 24_745)
    assert shape.triggers == (Trigger.LONG_LINE, Trigger.DENSE_LINES)


def test_the_stat_shortcut_is_exact_a_file_up_to_the_safe_size_can_never_be_over_the_bound() -> None:
    assert not measure(_one_line(PARSEABLE_UP_TO_BYTES)).too_large_to_parse
    assert measure(_one_line(PARSEABLE_UP_TO_BYTES + 1)).too_large_to_parse


def test_a_small_file_is_cleared_from_its_size_without_reading_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "small.js").write_bytes(_one_line(60_000))
    (tmp_path / "big.js").write_bytes(_one_line(668_777))
    reads: list[Path] = []
    original = Path.read_bytes

    def counting(self: Path) -> bytes:
        reads.append(self)
        return original(self)

    monkeypatch.setattr(Path, "read_bytes", counting)

    assert refusal_of(tmp_path, "small.js") is None
    assert reads == []
    assert refusal_of(tmp_path, "big.js").startswith("too large to parse")
    assert [path.name for path in reads] == ["big.js"]


def test_line_lengths_are_measured_in_bytes_so_multibyte_text_errs_on_the_safe_side() -> None:
    assert measure(("é" * 1_000).encode()).longest_line == 2_000


def test_a_refusal_counts_the_longest_line_in_bytes_as_measured() -> None:
    line = ("名();" * 12_000).encode()

    reason = measure(line).refusal

    assert reason is not None and reason.endswith("longest line 72,000 bytes")
