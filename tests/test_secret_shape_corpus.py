"""The analysis engine's own secret-masking corpus, shared so both maskers prove the same shapes.

Copied from analysis-engine ``enginepy/workflows/document_analysis/audit_readiness_test.py`` at
a0d5b0e4: each pair is a value and the engine's masked result. A word the engine hides must be
hidden here too, and a word it keeps must stay. The engine's own ``[REDACTED_*]`` re-masking cases
are left out, since this masker never writes those markers.
"""

import re

import pytest

from jev_navigator.judgments.secrets import SecretMasker

ENGINE_MASKED = [
    ("DB_PASSWORD=supersecret", "DB_PASSWORD=[REDACTED]"),
    ("MY_SECRET_TOKEN=abcdefghijklmnop", "MY_SECRET_TOKEN=[REDACTED]"),
    ("the_password=hunter2", "the_password=[REDACTED]"),
    ("export DB_PASSWORD=supersecret", "export DB_PASSWORD=[REDACTED]"),
    ("AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", "AWS_SECRET_ACCESS_KEY=[REDACTED]"),
    ("STRIPE_SECRET_KEY=sk_live_abcdefghijklmnop", "STRIPE_SECRET_KEY=[REDACTED]"),
    ("SECRET_KEY=abcdefghijklmnop", "SECRET_KEY=[REDACTED]"),
    ('password="correct horse battery staple"', "password=[REDACTED]"),
    ('DB_PASSWORD="correct horse battery staple"', "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD='correct horse battery staple'", "DB_PASSWORD=[REDACTED]"),
    ('DB_PASSWORD="correct \\"horse\\" battery"', "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD='correct \\'horse\\' battery'", "DB_PASSWORD=[REDACTED]"),
    ('DB_PASSWORD="unterminated secret', "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD='unterminated secret", "DB_PASSWORD=[REDACTED]"),
    ('DB_PASSWORD="unterminated secret\\', "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD='unterminated secret\\", "DB_PASSWORD=[REDACTED]"),
    ('DB_PASSWORD="abc', "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD='abc", "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD=correct\\ horse\\ battery", "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD: correct horse battery", "DB_PASSWORD: [REDACTED]"),
    ("DB_PASSWORD=alpha,beta", "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD=alpha}beta", "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD=alpha]beta", "DB_PASSWORD=[REDACTED]"),
    ('DB_PASSWORD=alpha"beta"', "DB_PASSWORD=[REDACTED]"),
    ('DB_PASSWORD="alpha"beta', "DB_PASSWORD=[REDACTED]"),
    ('DB_PASSWORD="alpha",beta', "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD='alpha'}beta", "DB_PASSWORD=[REDACTED]"),
    ("DB_PASSWORD=user@example.com", "DB_PASSWORD=[REDACTED]"),
    ("MY_SECRET_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz", "MY_SECRET_TOKEN=[REDACTED]"),
    (
        "private_key=-----BEGIN PRIVATE KEY-----\nabcdefghi\n-----END PRIVATE KEY-----",
        "private_key=[REDACTED]",
    ),
    ('{DB_PASSWORD: "secret", PUBLIC: visible}', "{DB_PASSWORD: [REDACTED], PUBLIC: visible}"),
    ('DB_PASSWORD: "alpha"beta', "DB_PASSWORD: [REDACTED]"),
    ("DB_PASSWORD: 'alpha'beta", "DB_PASSWORD: [REDACTED]"),
    ("DB_PASSWORD=secret\nPUBLIC=visible", "DB_PASSWORD=[REDACTED]\nPUBLIC=visible"),
    ('DB_PASSWORD="secret" PUBLIC=visible', "DB_PASSWORD=[REDACTED] PUBLIC=visible"),
    ("password: |\n  correct horse battery staple\nPUBLIC: visible", "password: [REDACTED]\nPUBLIC: visible"),
    (
        "  DB_PASSWORD: >-\n    line one\n    line two\n  PUBLIC: visible",
        "  DB_PASSWORD: [REDACTED]\n  PUBLIC: visible",
    ),
    ('DB_PASSWORD="line one\nline two"\nPUBLIC=visible', "DB_PASSWORD=[REDACTED]\nPUBLIC=visible"),
    ('DB_PASSWORD="line one\\\nline two"\nPUBLIC=visible', "DB_PASSWORD=[REDACTED]\nPUBLIC=visible"),
    ("DB_PASSWORD='line one\\\nline two'\nPUBLIC=visible", "DB_PASSWORD=[REDACTED]\nPUBLIC=visible"),
    (
        "- password: |\n    correct horse battery staple\n  public: visible",
        "- password: [REDACTED]\n  public: visible",
    ),
    (
        "items:\n  - DB_PASSWORD: >-\n      line one\n      line two\n  - PUBLIC: visible",
        "items:\n  - DB_PASSWORD: [REDACTED]\n  - PUBLIC: visible",
    ),
    (
        "password: &credential |\n  correct horse battery staple\npublic: visible",
        "password: [REDACTED]\npublic: visible",
    ),
    (
        "password: !<tag:yaml.org,2002:str> |\n  secret value\npublic: visible",
        "password: [REDACTED]\npublic: visible",
    ),
    (
        "- password: !!str >-\n    correct horse battery staple\n  public: visible",
        "- password: [REDACTED]\n  public: visible",
    ),
    (
        "db.password: |\n  correct horse battery staple\npublic: visible",
        "db.password: [REDACTED]\npublic: visible",
    ),
    ("password: correct horse\n  battery staple\npublic: visible", "password: [REDACTED]\npublic: visible"),
    (
        "- password: correct horse\n    battery staple\n  public: visible",
        "- password: [REDACTED]\n  public: visible",
    ),
    (
        'password: &credential "line one\n  secret value"\npublic: visible',
        "password: [REDACTED]\npublic: visible",
    ),
    ("password: !!str 'line one\n  secret value'\npublic: visible", "password: [REDACTED]\npublic: visible"),
    ("password: ! |\n  secret value\npublic: visible", "password: [REDACTED]\npublic: visible"),
    ("password: &credential ! |\n  secret value\npublic: visible", "password: [REDACTED]\npublic: visible"),
    ("DB_PASSWORD=correct\\\nhorse\\\nbattery\nPUBLIC=visible", "DB_PASSWORD=[REDACTED]\nPUBLIC=visible"),
    ("export DB_PASSWORD=correct\\\nhorse\nPUBLIC=visible", "export DB_PASSWORD=[REDACTED]\nPUBLIC=visible"),
    (
        "DB_PASSWORD=secret\\\\\nPUBLIC=visible\nNEXT=visible",
        "DB_PASSWORD=[REDACTED]\nPUBLIC=visible\nNEXT=visible",
    ),
    ("password:\n  token: inner-secret\npublic: visible", "password: [REDACTED]\npublic: visible"),
    ("password:\npublic: visible", "password: [REDACTED]\npublic: visible"),
    (
        '{\n  "password":\n    "secret",\n  "public": "visible"\n}',
        '{\n  "password": [REDACTED],\n  "public": "visible"\n}',
    ),
    ("password: {}", "password: [REDACTED]"),
    ("password: []", "password: [REDACTED]"),
    (
        '{"password": {"part": "secret"}, "public": "visible"}',
        '{"password": [REDACTED], "public": "visible"}',
    ),
    (
        '{"password": [{"part": "secret"}], "public": "visible"}',
        '{"password": [REDACTED], "public": "visible"}',
    ),
    (
        "password: {phrase: can't stop, nested: [one, two]}\npublic: visible",
        "password: [REDACTED]\npublic: visible",
    ),
    (
        "password: {phrase: Well 'tis fine, nested: [one, two]}\npublic: visible",
        "password: [REDACTED]\npublic: visible",
    ),
    (
        'password: {phrase: He said "hello today, nested: [one, two]}\npublic: visible',
        "password: [REDACTED]\npublic: visible",
    ),
    (
        'password: {phrase: &a "brace } text", nested: [one, two]}\npublic: visible',
        "password: [REDACTED]\npublic: visible",
    ),
    (
        'password: {phrase: !!str "brace } text", nested: [one, two]}\npublic: visible',
        "password: [REDACTED]\npublic: visible",
    ),
    (
        "password: {phrase: !<tag:yaml.org,2002:str> 'bracket ] text', nested: [one, two]}\npublic: visible",
        "password: [REDACTED]\npublic: visible",
    ),
    ("password: {part: ! &a 'secret ] } value'}\npublic: visible", "password: [REDACTED]\npublic: visible"),
    (
        'password: {? "key } text": secret, nested: [one, two]}\npublic: visible',
        "password: [REDACTED]\npublic: visible",
    ),
    (
        "password: {? &a 'key ] }': secret, nested: [one, two]}\npublic: visible",
        "password: [REDACTED]\npublic: visible",
    ),
]

# In a code file an identifier under a secret-named key reads as code (``{ id_token: token }``), so these
# stay; the first one holds identifiers only once its comment is set aside. In config text (a config file,
# or text from no file) the line-shaped ones are values, as the Engine reads them.
IDENTIFIER_VALUES = [
    (
        "password: {phrase: secret, # } and 'quotes' stay in the comment\n"
        "  nested: [one, two]\n}\npublic: visible",
        "password: [REDACTED]\npublic: visible",
    ),
    ("{DB_PASSWORD: secret, PUBLIC: visible}", "{DB_PASSWORD: [REDACTED], PUBLIC: visible}"),
    ('{"password": secret, "public": "visible"}', '{"password": [REDACTED], "public": "visible"}'),
    ('DB_PASSWORD: alpha,"beta":gamma', "DB_PASSWORD: [REDACTED]"),
    ("DB_PASSWORD: alpha,'beta':gamma", "DB_PASSWORD: [REDACTED]"),
    ("DB_PASSWORD: alpha,beta=gamma", "DB_PASSWORD: [REDACTED]"),
    ("DB_PASSWORD: alpha, beta:gamma", "DB_PASSWORD: [REDACTED]"),
    ('{"password": secret,"public":"visible"}', '{"password": [REDACTED],"public":"visible"}'),
]


def _words(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]{3,}", text)) - {"REDACTED", "MASKED"}


@pytest.mark.parametrize("value, engine_result", ENGINE_MASKED)
def test_every_word_the_engine_hides_is_hidden_and_every_word_it_keeps_stays(
    value: str, engine_result: str
) -> None:
    # Act
    masked = SecretMasker().mask(value)

    # Assert
    assert _words(masked) & (_words(value) - _words(engine_result)) == set()
    assert _words(engine_result) <= _words(masked)


CODE_PATH = "src/settings.ts"
CONFIG_LINES = [(value, result) for value, result in IDENTIFIER_VALUES if value.startswith("DB_PASSWORD: ")]


@pytest.mark.parametrize("value, engine_result", IDENTIFIER_VALUES)
def test_an_identifier_under_a_secret_named_key_stays_code(value: str, engine_result: str) -> None:
    # Act
    masked = SecretMasker().mask(value, CODE_PATH)

    # Assert
    assert masked == value


@pytest.mark.parametrize("value, engine_result", CONFIG_LINES)
def test_an_unquoted_config_value_hides_every_word_the_engine_hides(value: str, engine_result: str) -> None:
    # Act
    masked = SecretMasker().mask(value)

    # Assert
    assert _words(masked) & (_words(value) - _words(engine_result)) == set()
