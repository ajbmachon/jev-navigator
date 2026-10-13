"""The uses of the name a value on one line is assigned to, from a real git index."""

from git_repos import commit_files

from jev_navigator.index.code_index import CodeIndex
from jev_navigator.selection.value_uses import VALUE_USES, ValueUsesConfiguration, bound_name

SETTINGS = 'API_KEY = "value"\n' + "".join(f"line_{n} = {n}\n" for n in range(20)) + "send(API_KEY)\n"


def test_uses_come_from_the_anchor_file_then_other_files_then_tests_outside_the_shown_lines(tmp_path):
    commit_files(
        tmp_path,
        {
            "app/settings.py": SETTINGS.replace("line_3 = 3", "check(API_KEY)"),
            "app/client.py": "from app.settings import API_KEY\n",
            "docs/setup.md": "Set API_KEY before you start.\n",
            "tests/test_client.py": "patch('app.settings.API_KEY', 'fake')\n",
            "app/other.py": "MY_API_KEY_HOLDER = 1\n",
        },
    )

    uses = VALUE_USES.find(CodeIndex.from_git(tmp_path), "app/settings.py", 1, len('API_KEY = "'))

    assert uses.name == "API_KEY" and uses.found == 4
    assert [(hit.file, hit.line) for hit in uses.hits] == [
        ("app/settings.py", 22),
        ("app/client.py", 1),
        ("docs/setup.md", 1),
        ("tests/test_client.py", 1),
    ]


def test_a_block_local_or_too_common_name_stays_in_its_file_and_a_bare_value_has_no_name(tmp_path):
    commit_files(
        tmp_path,
        {
            "worker.py": "def run():\n    token = 'value'\n"
            + "".join(f"    x_{n} = {n}\n" for n in range(15))
            + "    return token\n",
            "other.py": "token = None\n",
            "keys.pem": "value\n",
        },
    )
    index = CodeIndex.from_git(tmp_path)

    local = VALUE_USES.find(index, "worker.py", 2, len("    token = '"))
    common = ValueUsesConfiguration(max_scope_hits=0).find(index, "other.py", 1, len("token = "))

    assert [(hit.file, hit.line) for hit in local.hits] == [("worker.py", 18)] and local.found == 1
    assert common.found == 0
    assert VALUE_USES.find(index, "keys.pem", 1, 0).name is None


def test_the_bound_name_is_the_key_variable_or_argument_before_the_value():
    assert bound_name('    "api_key": "') == "api_key"
    assert bound_name("password=") == "password"
    assert bound_name("export TOKEN=") == "TOKEN"
    assert bound_name("id = ") is None
    assert bound_name("print(") is None
