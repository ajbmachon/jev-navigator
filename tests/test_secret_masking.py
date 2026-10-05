import re
import time
from collections import Counter

import pytest

from jev_navigator.judgments.secrets import (
    HIGH_ENTROPY_MIN_CHARS,
    MASK,
    TOKEN_CHARACTER_CLASS,
    SecretInRequestError,
    SecretMasker,
    SecretScanner,
    is_high_entropy,
    mask_by_content,
    mask_request,
    refuse_if_secret,
)


def test_repeated_request_text_is_scanned_once_without_retaining_other_requests(monkeypatch):
    masker = SecretMasker()
    mask = masker.mask
    discover = masker.masked_values
    masked = Counter()
    discovered = Counter()

    def observe_mask(self, text, path=None):
        masked[text] += 1
        return mask(text, path)

    def observe_discovery(self, text, path=None):
        discovered[text] += 1
        return discover(text, path)

    monkeypatch.setattr(SecretMasker, "mask", observe_mask)
    monkeypatch.setattr(SecretMasker, "masked_values", observe_discovery)
    reference = 'send("order-hook-4f7a1c")'
    value = {
        "assignment": 'WEBHOOK_TOKEN = "order-hook-4f7a1c"',
        "rows": [{"code": reference} for _ in range(200)],
    }

    result = mask_by_content(value, masker)

    assert result == {
        "assignment": 'WEBHOOK_TOKEN = "[MASKED]"',
        "rows": [{"code": 'send("[MASKED]")'} for _ in range(200)],
    }
    assert discovered[reference] == 1
    assert masked[reference] == 1
    # Without the assignment this ordinary string is not secret-shaped. The previous request's
    # discovered values must not survive as hidden state in a later masking operation.
    assert mask_by_content({"code": reference}, masker) == {"code": reference}


SLACK_TOKEN = "xoxb" + "-123456789012-abcdefghijklmnop"
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
PRIVATE_KEY = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEAx9k2\n-----END RSA PRIVATE KEY-----"

SECRET_VALUES = {
    "private key": (f"const key = `{PRIVATE_KEY}`;", "MIIEowIBAAKCAQEAx9k2"),
    "JWT": (
        'const t = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.'
        'dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U";',
        "dozjgNryP4J3jVmNHl0w5N",
    ),
    "GitHub token": (f'auth: "{GITHUB_TOKEN}"', GITHUB_TOKEN),
    "GitHub fine-grained token": (
        'auth: "github_pat_11ABCDEFG0abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP"',
        "github_pat_11ABCDEFG0",
    ),
    "GitLab token": ('const t = "glpat-abcdefghijklmnopqrst";', "glpat-abcdefghijklmnopqrst"),
    "npm token": (
        'const t = "npm_abcdefghijklmnopqrstuvwxyz0123456789";',
        "npm_abcdefghijklmnopqrstuvwxyz0123456789",
    ),
    "OpenAI key": (
        'const k = "sk-proj-abcdefghijklmnopqrstuvwxyz012345";',
        "sk-proj-abcdefghijklmnopqrstuvwxyz012345",
    ),
    "Slack token": (f'const s = "{SLACK_TOKEN}";', SLACK_TOKEN),
    "AWS access key id": ('const id = "AKIAIOSFODNN7EXAMPLE";', "AKIAIOSFODNN7EXAMPLE"),
    "Google API key": (
        'const g = "AIzaSyA1234567890abcdefghijklmnopqrstuv";',
        "AIzaSyA1234567890abcdefghijklmnopqrstuv",
    ),
    "Bearer header": ('headers = {"Authorization": "Bearer abcdefghijklmnop1234"}', "abcdefghijklmnop1234"),
    "quoted secret in JavaScript": (
        'const options = { secret: "s3cr3t-session-value" };',
        "s3cr3t-session-value",
    ),
    "quoted password in Python": ("password = 'hunter2-hunter2'", "hunter2-hunter2"),
    "short quoted password": ("password = 'hunter2'", "hunter2"),
    "quoted JSON key": ('"token": "abcd1234efgh5678"', "abcd1234efgh5678"),
    "bare YAML value": ("api_key: plain-yaml-secret-value", "plain-yaml-secret-value"),
    "env-file value": ("DATABASE_PASSWORD=Pr0dPassw0rd2024", "Pr0dPassw0rd2024"),
    "literal fallback after a reference": (
        'secret: process.env.SESSION_SECRET || "fallback-secret"',
        "fallback-secret",
    ),
    "template literal": ("token = `template-secret-value`", "template-secret-value"),
    "word argument to a signing call": ('token = sign(payload, "hunter2")', "hunter2"),
    "literal argument to a secret-named call": (
        "secret: getSecret('sk-live-signing-0042')",
        "sk-live-signing-0042",
    ),
    "bcrypt password hash": (
        'password_hash = "$2b$12$KIXQJpZ8sWm3eVt7Lq9u0OaBcDeFgHiJkLmNoPqRsTuVwXyZ01234"',
        "KIXQJpZ8sWm3eVt7Lq9u0OaBcDeFgHiJkLmNoPqRsTuVwXyZ01234",
    ),
    "single-quoted value starting with a dollar sign": ("api_key = '$ecretValue123'", "ecretValue123"),
    "value with an interpolation inside": ('jwt_secret = "abcd${x}1234efgh5678"', "1234efgh5678"),
    "value under a key that names an environment": ('SECRET_ENV = "prod-hunter2-xyz"', "prod-hunter2-xyz"),
    "value under a key that names a label": ('password_label = "Sup3rS3cret!"', "Sup3rS3cret!"),
    "value under a key that names a file": (
        'client_secret_file = "s3cr3t-value-not-a-file"',
        "s3cr3t-value-not-a-file",
    ),
    "value under a key that names a header": (
        'auth_token_header = "abcd1234efgh5678ijkl"',
        "abcd1234efgh5678ijkl",
    ),
    "token in a URL query": (
        'token_url = "https://example.test/cb?token=abcd1234efgh5678"',
        "abcd1234efgh5678",
    ),
    "password in a URL": ('DATABASE_URL = "postgres://app:Pr0dPassw0rd@db:5432/app"', "Pr0dPassw0rd"),
    "value under a password-hash key": (
        'password_hash = "pbkdf2-sha256-600000-abcdef"',
        "pbkdf2-sha256-600000-abcdef",
    ),
    "unquoted hex key under a lower-case key": ("secret_key_base=" + "4f" * 64, "4f" * 64),
    "unquoted generated value": ("webhook_secret_v1=whsec_" + "a1B2" * 8, "a1B2" * 8),
    "high-entropy value under an ordinary name": (
        'const signingKey = "Zq8vT2mN4xR7pL1wK9sD3fH6";',
        "Zq8vT2mN4xR7pL1wK9sD3fH6",
    ),
}

CODE_REFERENCES = [
    "secret: process.env.BETTER_AUTH_SECRET,",
    "token: recipient.token, });",
    "{ id_token: token, id }",
    "const options = { secret: process.env.SESSION_SECRET };",
    'password = os.environ["DB_PASSWORD"]',
    'api_key = settings.get("api_key")',
    '  "token": request.headers.authorization,',
    "const token = await getAccessToken(user);",
    "if (user.email === 'admin@example.com') return 10.0.0.1;",
    "token: Record<string, string>;",
    'password: "${DB_PASSWORD}"',
    "api_key: ${API_KEY}",
    "--token-size: 12px;",
    "signal('SIGTERM')",
    "Returns a Bearer token for the user.",
    "JEV_STATE_TOKEN_LIMIT = 32_000",
    'refuse_if_secret({"prompt": prompt}, {}, scanner)',
    "never print credentials (environment or `~/.config/jvn/env`).",
    "'@aws-sdk/credential-provider-ini': 3.973.15",
    "secretName: heedvane-observability-runtime",
    'SECRETS_DIRECTORY = "secrets"',
    'credentialsMountPath: "/var/run/secrets/google",',
    'if [[ -z "${SLACK_BOT_TOKEN:-}" ]]; then',
    "_render_reports_block(reports, token_budget=...)",
    'secretAnnotation(kind, "name")',
    'requireSecretEnvironment(config, "RUNNER_AUTH_TOKEN", "engine-secrets", "runner-auth-token")',
    'my_token = "${TOKEN}"',
    'DATABASE_URL = "postgres://app:${DB_PASSWORD}@db:5432/app"',
    'token_url = "https://example.test/oauth/token"',
    'client_secret_file = "/run/secrets/client_secret.json"',
    'SECRET_ENV = "production"',
    'password_label = "Password"',
    "packages/trpc/server/api-token-router/create-api-token.ts:9-43",
    "scripts/check-github-oauth-credentials.mjs:25-35",
    "getUserFromSessionToken(context, queryInfo, 'user.', false)",
    'return createHmac("sha256", key)',
    "crypto.createHmac('sha1', key)",
    "MAX_TOKENS_MARKER = 'max_tokens_exceeded'",
    " *   REQUESTY_API_KEY=... REQUESTY_RECEIPT=/absolute/path/receipt.json \\",
    "const USAGE = 'Usage: REQUESTY_API_KEY=<credential> '",
]


@pytest.mark.parametrize("code, value", SECRET_VALUES.values(), ids=SECRET_VALUES.keys())
def test_secret_values_are_masked(code: str, value: str) -> None:
    # Act
    masked = SecretMasker().mask(code)

    # Assert
    assert value not in masked
    assert MASK in masked


@pytest.mark.parametrize("code, value", SECRET_VALUES.values(), ids=SECRET_VALUES.keys())
def test_the_scanner_reports_every_secret_value_the_masker_hides(code: str, value: str) -> None:
    # Act
    findings = SecretScanner().findings(code)

    # Assert
    assert findings
    assert SecretScanner().findings(SecretMasker().mask(code)) == []


CODE_PATH = "src/app/module.ts"


@pytest.mark.parametrize("code", CODE_REFERENCES)
def test_code_references_reach_jev_byte_identical(code: str) -> None:
    # Act
    masked = SecretMasker().mask(code, CODE_PATH)

    # Assert
    assert masked == code
    assert SecretScanner().findings(code, CODE_PATH) == []


def test_only_the_value_is_masked_when_the_key_has_the_same_text() -> None:
    # Act
    masked = SecretMasker().mask('password: "password1234"\nexport const password = "password"')

    # Assert
    assert masked == 'password: "[MASKED]"\nexport const password = "[MASKED]"'


def test_a_short_value_is_masked_where_it_stands_and_nowhere_else_in_the_request() -> None:
    # Arrange
    state = {
        "assignment": "password = 'test'",
        "path": "src/test/login_test.py",
        "test": "a key that names the short value",
    }

    # Act
    masked, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked, questions, SecretScanner(), values)

    # Assert
    assert masked == {
        "assignment": "password = '[MASKED]'",
        "path": "src/test/login_test.py",
        "test": "a key that names the short value",
    }
    assert values == frozenset()


def test_a_long_bare_value_is_masked_everywhere_in_the_request() -> None:
    # Act
    masked = mask_by_content(
        {"config": "api_key: plain-yaml-secret-value", "log": "sent plain-yaml-secret-value upstream"},
        SecretMasker(),
    )

    # Assert
    assert masked == {"config": "api_key: [MASKED]", "log": "sent [MASKED] upstream"}


LONG_LINES = {
    "dash chain": "a-" * 35_000,
    "SVG path data": 'd="M' + "".join(f"{i % 97}.{i % 13}-" for i in range(12_000)) + '"',
    "upper-case run": "A_" * 35_000 + "=x",
    "dotted secret-word run": "token." * 12_000 + "x",
    "secret-named call run": "getToken" * 9_000 + "(x)",
    "assignment pairs": "a=b " * 8_000,
    "inline SVG attributes": "<svg " + 'x="1" y="2" fill="none" ' * 2_700 + "/>",
}


@pytest.mark.parametrize("text", LONG_LINES.values(), ids=LONG_LINES.keys())
def test_masking_a_long_line_takes_time_linear_in_its_length(text: str) -> None:
    # Act
    started = time.perf_counter()
    SecretMasker().mask(text)
    SecretScanner().findings(text)

    # Assert
    assert time.perf_counter() - started < 1.0


REPEATED_PAIRS = {"assignment pairs": "a=b ", "SVG attributes": 'x="1" '}


@pytest.mark.parametrize("pair", REPEATED_PAIRS.values(), ids=REPEATED_PAIRS.keys())
def test_masking_time_grows_linearly_as_a_line_of_pairs_doubles(pair: str) -> None:
    # Arrange
    lengths = [8_000, 16_000, 32_000, 64_000]

    # Act
    seconds = [_fastest_mask_seconds(pair * (length // len(pair))) for length in lengths]

    # Assert
    assert seconds[-1] < 24 * max(seconds[0], 0.001), seconds
    assert seconds[-1] < 0.5, seconds


def _fastest_mask_seconds(text: str) -> float:
    timings = []
    for _ in range(3):
        started = time.perf_counter()
        SecretMasker().mask(text)
        timings.append(time.perf_counter() - started)
    return min(timings)


SUFFIXED_SECRET_KEYS = [
    "SECRET_KEY_BASE", "API_KEY_2", "api_key_v2", "apiKey2", "TOKEN_GITHUB", "DB_PASSWORD_PROD",
    "dbPasswordProd", "STRIPE_SECRET_LIVE", "password1", "PASSWORD_CONFIRMATION", "access_token_secret",
    "client_secret_value", "refresh_token_old", "MYSQL_ROOT_PASSWORD", "mysql_password_root",
    "secretAccessKeyId", "GH_TOKEN_RO", "webhook_secret_v1", "PASSWORD_SALT", "pwd_admin", "credentials_json",
    "db_pass", "userPwd", "credentials",
]  # fmt: skip


@pytest.mark.parametrize("key", SUFFIXED_SECRET_KEYS)
@pytest.mark.parametrize(
    "shape", ['{key} = "{value}"', "{key}: '{value}'", "{key}={value}", '"{key}": "{value}",']
)
def test_a_secret_word_anywhere_in_the_key_makes_it_secret(key: str, shape: str) -> None:
    # Act
    masked = SecretMasker().mask(shape.format(key=key, value="hunter2-hunter2x"))

    # Assert
    assert "hunter2-hunter2x" not in masked


@pytest.mark.parametrize(
    "code",
    [
        'bypass = "allow-all-traffic"',
        'compass = "north-north-west"',
        'passport = "travel-docs-only"',
        "max_tokens = 4096",
        'MAX_TOKENS_MARKER = "max_tokens_exceeded"',
        'tokenizer = "cl100k_base"',
        "packages/trpc/server/api-token-router/create-api-token.ts:9-43",
        'credentialsMountPath: "/var/run/secrets/google",',
    ],
)
def test_a_secret_word_inside_another_word_is_not_a_secret_key(code: str) -> None:
    # Act
    masked = SecretMasker().mask(code)

    # Assert
    assert masked == code


@pytest.mark.parametrize(
    "line",
    [
        'DB_PASSWORD_PROD = "hunter2"',
        'GH_TOKEN_RO: "s3cr3t"',
        'GH_TOKEN_RO = "correct horse battery staple"',
        'PASSWORD_ERROR = "hunter2"',
        "SECRET_KEY_BASE=abc",
        '"credentials_json": "{}x",',
        'password_hash = "pw"',
    ],
)
def test_a_short_value_under_a_suffixed_secret_key_is_masked(line: str) -> None:
    # Act
    masked = SecretMasker().mask(line)

    # Assert
    assert "[MASKED]" in masked


@pytest.mark.parametrize(
    "code",
    [
        'INSPECTION_STATUS = Object.freeze({ PASS: "PASS", FAIL: "FAIL" });',
        'FAIL = "fail"',
        'PASSWORD_ERROR = "Password must be at least 8 characters."',
        'TOKEN_HELP_TEXT = "Paste the token from your settings page."',
        'Token: "token"',
        'TOKEN_TYPE = "type"',
        'DB_PASSWORD_PROD: "DB_PASSWORD"',
        'GH_TOKEN_RO = "/run/secrets/gh-token"',
        'SECRET_KEY_BASE = "https://vault.example.com/base"',
        'SECRET_FILE_RULE = "secret-file"',
        'password_prefix = "pw_"',
        'token_count = "12"',
        "CREDENTIAL_PATTERNS = [/gh_x/g]",
    ],
)
def test_a_value_its_key_already_shows_or_a_reference_under_a_suffixed_key_is_kept(code: str) -> None:
    # Act
    masked = SecretMasker().mask(code)

    # Assert
    assert masked == code


def test_a_credential_under_a_naming_key_is_masked() -> None:
    # Act
    masked = SecretMasker().mask('secretAccessKeyId = "a8f9e0d1c2b3a4f5"')

    # Assert
    assert "a8f9e0d1c2b3a4f5" not in masked


COMPOSE = "services:\n  db:\n    environment:\n      POSTGRES_PASSWORD: example\n      POSTGRES_USER: app\n"


@pytest.mark.parametrize(
    ("file", "code", "secret"),
    [
        ("docker-compose.yml", COMPOSE, "example"),
        ("deploy/app.env", "DB_PASSWORD=example\n", "example"),
        (".env.local", "API_TOKEN: hunter2\n", "hunter2"),
        ("config/app.ini", "[db]\npassword = example\n", "example"),
        ("Dockerfile", "ENV DB_PASSWORD=example\n", "example"),
        ("pyproject.toml", "api_token = hunter2\n", "hunter2"),
    ],
)
def test_an_unquoted_value_under_a_secret_key_in_a_config_file_is_masked(
    file: str, code: str, secret: str
) -> None:
    # Arrange
    state = {"slice": {"file": file, "lines": "1-5", "code": code}}

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert secret not in masked_state["slice"]["code"]


@pytest.mark.parametrize(
    ("file", "code"),
    [
        ("app/auth.py", "def login(token: str) -> None:\n    password: str = token\n"),
        ("web/form.ts", "const form = { password: hunter2 };\n"),
        (
            "docker-compose.yml",
            "environment:\n  POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}\n  DB_TOKEN: $DB_TOKEN\n",
        ),
        ("values.yaml", "auth:\n  password:\n  token: null\n"),
        ("pod.yaml", "spec:\n  automountServiceAccountToken: false\n"),
        (".github/workflows/ci.yml", "env:\n  API_TOKEN: ${{ secrets.API_TOKEN }}\n"),
    ],
)
def test_code_and_config_references_keep_their_unquoted_values(file: str, code: str) -> None:
    # Arrange
    state = {"slice": {"file": file, "lines": "1-3", "code": code}}

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert masked_state == state


def test_text_without_a_file_is_read_as_config() -> None:
    # Act
    masked = SecretMasker().mask("POSTGRES_PASSWORD: example")

    # Assert
    assert masked == "POSTGRES_PASSWORD: [MASKED]"


def test_the_final_scan_reads_a_slice_as_its_file_type() -> None:
    # Arrange
    code_slice = {"slice": {"file": "app/form.ts", "code": "const form = {\n  token: abc123,\n};\n"}}
    config_slice = {"slice": {"file": "app.yaml", "code": "token: abc123\n"}}

    # Act
    refuse_if_secret(code_slice, {}, SecretScanner())

    # Assert
    with pytest.raises(SecretInRequestError):
        refuse_if_secret(config_slice, {}, SecretScanner())


@pytest.mark.parametrize(
    "line",
    [
        "API_TOKEN=hunter2xyz OUT=web/gen.js node build.mjs",
        "\tAPI_TOKEN=hunter2xyz OUT=web/gen.js $(MAKE) gen",
        "run: API_TOKEN=hunter2xyz npm test",
        "RUN API_TOKEN=hunter2xyz ./build.sh && echo done",
        "cd web && API_TOKEN=hunter2xyz make",
    ],
)
@pytest.mark.parametrize("path", ["Makefile", "build.sh", ".github/workflows/ci.yml", "Dockerfile", None])
def test_an_assignment_followed_by_more_of_the_line_is_masked(line: str, path: str | None) -> None:
    # Act
    masked = SecretMasker().mask(line, path)

    # Assert
    assert "hunter2xyz" not in masked
    assert "gen.js" in masked or "gen.js" not in line


def test_the_randomness_check_is_public_and_names_its_token_shape() -> None:
    # Arrange
    token = "aZ3kQ9pL2xV7mN4bR8tY1wE6"

    # Act
    random_enough = is_high_entropy(token)

    # Assert
    assert random_enough and not is_high_entropy("a" * 24)
    assert len(token) >= HIGH_ENTROPY_MIN_CHARS
    assert re.fullmatch(f"{TOKEN_CHARACTER_CLASS}+", token)
