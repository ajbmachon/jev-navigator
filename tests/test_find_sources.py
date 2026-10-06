from __future__ import annotations

import json
from pathlib import Path

from find_moves import FROZEN, moves_index, sent_requests


def test_find_sends_the_requests_it_sent_before_its_moves_became_sources(tmp_path: Path) -> None:
    # Arrange: every move lists a place in this repository, and the search opens every place it reaches
    index = moves_index(tmp_path)
    frozen = json.loads(FROZEN.read_text())

    # Act
    requests = sent_requests(index)

    # Assert
    assert requests == frozen
