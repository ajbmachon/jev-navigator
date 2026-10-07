from pathlib import Path

import pytest
from shop_search import shop_index

from jev_navigator.plan_outline import outline_lines


def test_outline_binds_actual_declarations_imports_and_text_blocks_without_bodies(tmp_path: Path):
    index = shop_index(
        tmp_path,
        {
            "entry.py": "from helper import check\n\ndef run():\n    return check('SECRET_BODY')\n",
            "helper.py": "def check(value):\n    return value\n",
            "docs.md": "# Policy\nKeep the limit.\n",
        },
    )
    outlined = "".join(outline_lines(index, ("entry.py", "docs.md")))
    assert "entry.py\n  run:3-4\n  ->helper.py\n" in outlined
    assert "docs.md\n  Policy:1-2\n" in outlined
    assert "SECRET_BODY" not in outlined
    with pytest.raises(ValueError, match="not a file"):
        list(outline_lines(index, ("invented.py",)))
