import json
import re
import time
from collections import Counter
from pathlib import Path

import pytest

from jev_navigator.judgments.questions import Check, Criterion
from jev_navigator.judgments.secrets import (
    HIGH_ENTROPY_MIN_CHARS,
    MASK,
    TOKEN_CHARACTER_CLASS,
    SecretInRequestError,
    SecretMasker,
    SecretScanner,
    copy_pattern,
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
    "unterminated quoted value": ('password: "unterminated secret', "unterminated secret"),
    "camelCase secret key": ('const authToken = "hunter2";', "hunter2"),
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
    "bytes literal": ('secret_key = b"abcd1234efgh5678"', "abcd1234efgh5678"),
    "raw string literal": ('SECRET_KEY = r"abcd1234efgh5678"', "abcd1234efgh5678"),
    "triple-quoted literal": ('private_key = """abcd1234efgh5678"""', "abcd1234efgh5678"),
    "nested value a window cut before it closed": (
        'password: {"part": "s3cret-value", "next": ',
        "s3cret-value",
    ),
    "unquoted hex key under a lower-case key": ("secret_key_base=" + "4f" * 64, "4f" * 64),
    "short value under a suffixed key": ("DB_PASSWORD_PROD=hunter2", "hunter2"),
    "short quoted value under a suffixed key": ('SECRET_KEY_BASE: "s3cret"', "s3cret"),
    "dollar sign inside a password": ('password = "my$ecret"', "ecret"),
    "dollar sign inside a token": ('token: "a$b1234567"', "b1234567"),
    "unquoted generated value": ("webhook_secret_v1=whsec_" + "a1B2" * 8, "a1B2" * 8),
    "nested quoted words under a key that describes nothing": (
        "password: {\n  value: 'correct horse battery staple',\n}",
        "correct horse battery staple",
    ),
    "nested single word under a key that describes": ("password: {\n  hint: 'hunter22x',\n}", "hunter22x"),
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
    "        fencing_token=self.identity.fencing_token,",
    '        hub_token=token or "",',
    "          csrfToken={csrfToken}",
    'this.name = "RunAttemptExecutionClaimConflictError";',
    "secret = {path for file in cited if (path := repo_relative_path(file, repo)) is not None}",
    "# fixture paths are tagged secret: likely fixture by the scanner",
    "// Deprecated env token: an exact match resolves without a round-trip.",
    'RunsRestToken: { in: "header", name: "x-heedvane-runs-rest-token", type: "apiKey" },',
    "existingSecret: { encryptedSecret: Uint8Array; encryptionKeyVersion: number } | null;",
    "secret: {\n  name: AUTH_SECRET_NAME,\n"
    '  items: [{ key: "proxy.htpasswd", path: "proxy.htpasswd", mode: 0o440 }],\n},',
    "emailAndPassword: {\n  enabled: true,\n  // the bounds are the server's copy\n"
    "  minPasswordLength: PASSWORD_MIN_LENGTH,\n},",
    "volumes:\n  - name: runtime\n    secret:\n      secretName: observability-runtime\n"
    "      items:\n        - key: metrics-token\n          path: metrics_token\n",
    "secrets:\n  READ_TOKEN:\n    description: Read-only token for the exact checkout.\n"
    "    required: false\n",
    'WEBHOOK_SECRET="whsec_$(openssl rand -base64 32)"',
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
    'AND "cancellationEventLeaseToken" = $5',
    "// URL user-info credentials (`postgresql://user:password@host/db`) are masked",
    "Set `HEEDVANE_PROXY_URL=http://user:pass@proxy:3128` first.",
    "// here with ?token=… when valid",
    "clientSecret: `GITLAB_INTEGRATION_CLIENT_SECRET_${slug}`,",
    "const gitSecretName = `inv-${input.inventoryId}-${input.generation}-git`;",
    'need = isCredential(name) ? "must use valueFrom.secretKeyRef" : "is not an approved literal";',
    "return `read -rsp 'GitLab token: ' GITLAB_TOKEN && printf '\\n' && export GITLAB_TOKEN && ` +",
    'CREDENTIAL_PATTERNS = [\n  { label: "github-token", pattern: /gh_x/g },\n];',
    "secret-scan:\n  runs-on: ubuntu-latest\n  steps:\n    - name: Install pinned Gitleaks\n",
    'credentialsSourcePath: "/var/run/secrets/google/credentials.json",',
    "        token_budget=(",
    "        password=(",
    'export GOOGLE_APPLICATION_CREDENTIALS="$CI_TMP/google-adc.json"',
    "print(f\"GATE pass={c['gate_pass']} confidence={c.get('confidence')}\")",
    '"rawCredentialInherited": "GOOGLE_VERTEX_CREDENTIALS_JSON" in os.environ,',
    "clientSecret: `[MASKED]_${slug}`,",
    "secret-scan: run the scan nightly",
    " *   REQUESTY_API_KEY=... REQUESTY_RECEIPT=/absolute/path/receipt.json \\",
    "const USAGE = 'Usage: REQUESTY_API_KEY=<credential> '",
    "` -e HEEDVANE_ENROLLMENT_TOKEN=${shellQuote(input.enrollmentToken)}` +",
    "  ? `never cached (${row.prefixTokens ?? '?'}-token prefix, likely below)`",
    'lines = [line for line in values if line.startswith("DB_PASSWORD: ")]',
    "  restAPIKey: {\n    env: 'PARSE_SERVER_REST_API_KEY',\n    help: 'Key for REST calls',\n  },",
    "  proxyPassword: {\n    env: 'PARSE_SERVER_DATABASE_PROXY_PASSWORD',\n    help:\n"
    "      'The MongoDB driver option to configure a Socks5 proxy password when the proxy requires "
    "username/password authentication.',\n  },",
    "      password: {\n        descriptions: 'New password of the user',\n"
    "        type: new GraphQLNonNull(GraphQLString),\n      },",
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


def test_a_short_value_is_masked_wherever_it_stands_as_a_whole_word() -> None:
    # Arrange
    state = {
        "assignment": "password = 'test'",
        "path": "src/test/login_test.py",
        "note": "a test of the short value",
    }

    # Act
    masked, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked, questions, SecretScanner(), values)

    # Assert
    assert masked == {
        "assignment": "password = '[MASKED]'",
        "path": "src/[MASKED]/login_test.py",
        "note": "a [MASKED] of the short value",
    }
    assert values == frozenset({"test"})


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
    "unclosed flow values": "password: {" * 6_000,
    "unterminated quoted value": 'password: "' + "a" * 70_000,
    "plain value with a long space run": "password: a" + " " * 70_000 + "x",
    "flow value of many quoted pairs": "secret: {" + '"a": "b", ' * 7_000 + "}",
    "YAML block of many lines": "password:\n" + "  x: y\n" * 10_000,
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
    "db_pass", "db_passwd", "userPwd", "credentials",
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
        'GH_TOKEN_RO = process.env.GH_TOKEN ?? "dev"',
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


@pytest.mark.parametrize(
    ("text", "secret"),
    [
        ("DB_PASSWORD_PROD: |\n  hunter2\n  rest\n", "hunter2"),
        ("DB_PASSWORD_PROD: >-\n  hunter2\n", "hunter2"),
        ('DB_PASSWORD = "abc" + "hunter2"', "hunter2"),
        ('GH_TOKEN_RO = "abc" \\\n  "hunter2"', "hunter2"),
        ('$db_password = "abc" . "hunter2";', "hunter2"),
    ],
)
def test_a_value_that_belongs_to_a_secret_key_is_masked_whole(text: str, secret: str) -> None:
    # Act
    masked = SecretMasker().mask(text)

    # Assert
    assert secret not in masked


def test_a_nested_table_under_a_suffixed_key_judges_its_inner_keys() -> None:
    # Arrange
    text = "DB_PASSWORD_PROD:\n  user: app\n  host: db.internal\n"

    # Act
    masked = SecretMasker().mask(text)

    # Assert
    assert masked == text


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


STRIPE_LIVE = "sk_live_" + "4eC39HqLyjWDarjtT1zdp7dc"


@pytest.mark.parametrize(
    ("file", "code", "secret"),
    [
        (
            "tests/test_x.py",
            'items = [{"code": \'x = 1\\npassword = "hunter2hunter2"\'}]\n',
            "hunter2hunter2",
        ),
        (
            "scripts/env.mjs",
            f'  const lines = [\n    "STRIPE_SECRET_KEY={STRIPE_LIVE}",\n  ];\n',
            STRIPE_LIVE,
        ),
        ("app/pay.py", f"# live key {STRIPE_LIVE}\n", STRIPE_LIVE),
        ("scripts/env.mjs", '  const lines = [\n    "DB_PASSWORD=hunter2hunter2",\n  ];\n', "hunter2hunter2"),
        ("Makefile", "\tdocker run -e DB_PASSWORD=hunter2hunter2 app\n", "hunter2hunter2"),
        ("scripts/run.sh", "psql --password=hunter2hunter2 -h db\n", "hunter2hunter2"),
        ("scripts/run.sh", "deploy --api-token hunter2hunter2 --region eu\n", "hunter2hunter2"),
        ("scripts/run.sh", "deploy \\\n--api-token hunter2hunter2\n", "hunter2hunter2"),
    ],
)
def test_a_secret_the_base_masker_left_in_code_is_masked(file: str, code: str, secret: str) -> None:
    # Act
    masked_state, _, _ = mask_request({"slice": {"file": file, "code": code}}, {}, SecretMasker())

    # Assert
    assert secret not in masked_state["slice"]["code"]


def test_a_short_value_masked_at_its_key_is_masked_wherever_it_stands_as_a_word() -> None:
    # Arrange
    state = {
        "slice": {"file": "app/settings.py", "code": 'DB_PASSWORD_PROD = "hunter2"\n'},
        "candidates": [{"file": "app/db.py", "code": 'connect(user, "hunter2")\nlabel = "hunter2x"\n'}],
    }

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert masked_state["candidates"][0]["code"] == 'connect(user, "[MASKED]")\nlabel = "hunter2x"\n'


def test_a_short_number_masked_at_its_key_stays_elsewhere() -> None:
    # Arrange
    state = {
        "slice": {"file": "app/settings.ts", "code": 'CREDENTIAL_KEY_VERSION: "1.5",\n'},
        "other": {"file": "app/math.ts", "code": "const ratio = 1.5;\n"},
    }

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert masked_state["other"]["code"] == "const ratio = 1.5;\n"


def test_a_short_masked_value_inside_a_longer_word_does_not_refuse_the_request() -> None:
    # Arrange
    state, questions, masked = mask_request(
        {"slice": {"file": "app/a.py", "code": 'db_pass = "pwd1"\n'}, "pwd1_hint": "x"}, {}, SecretMasker()
    )

    # Act
    refuse_if_secret(state, questions, SecretScanner(), masked)

    # Assert
    assert "pwd1" in masked


def test_a_value_too_short_to_identify_a_secret_is_masked_only_at_its_key() -> None:
    # Arrange
    state = {
        "slice": {"file": "tests/fixture.ts", "code": 'const secret = "ghp_" + "x" * 36;\n'},
        "other": {"file": "src/loop.ts", "code": "for (const x of xs) { use(x); }\n"},
    }

    # Act
    masked_state, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert masked_state["slice"]["code"] == 'const secret = "[MASKED]" + "[MASKED]" * 36;\n'
    assert masked_state["other"]["code"] == "for (const x of xs) { use(x); }\n"


VALUE = "hunter2hunter2"


@pytest.mark.parametrize(
    ("file", "code"),
    [
        ("tests/test_x.py", f'files = {{"settings.py": \'WEBHOOK_TOKEN = "{VALUE}"\\n\'}}\n'),
        ("package.json", f'{{"scripts": {{"start": "DB_PASSWORD=\'{VALUE}\' node app.js"}}}}\n'),
        ("docker-compose.yml", f"    command: \"export API_TOKEN='{VALUE}' && run\"\n"),
        ("app/docs.py", f"HELP = 'set password = \"{VALUE}\" first'\n"),
        ("app/settings.py", f'DB_PASSWORD_PROD = """\n{VALUE}\n"""\n'),
        ("app/settings.py", "DB_PASSWORD = " + "'" * 3 + f"\n    {VALUE}\n" + "'" * 3 + "\n"),
        ("src/db.ts", f"const DB_PASSWORD = `\n  {VALUE}\n`;\n"),
        ("src/Db.java", f'String password = """\n    {VALUE}\n    """;\n'),
        ("config/app.toml", f'db_password = """\n{VALUE}"""\n'),
        ("app/settings.py", f'DB_PASSWORD = " {VALUE}"\n'),
        ("app/settings.py", f'SECRET_KEY = (\n    "{VALUE}"\n)\n'),
        ("app/settings.py", f'SECRET_KEY = (\n    "first-part-x"\n    "{VALUE}"\n)\n'),
    ],
)
def test_a_secret_inside_another_string_or_across_lines_is_masked(file: str, code: str) -> None:
    # Act
    masked_state, _, _ = mask_request({"slice": {"file": file, "code": code}}, {}, SecretMasker())

    # Assert
    assert VALUE not in masked_state["slice"]["code"]


def test_masking_keeps_the_quotes_around_a_hidden_value() -> None:
    # Act
    masked = SecretMasker().mask(f"assert line == 'WEBHOOK_TOKEN = \"{VALUE}\"'", "tests/test_x.py")

    # Assert
    assert masked == "assert line == 'WEBHOOK_TOKEN = \"[MASKED]\"'"


def test_a_value_hidden_at_its_key_is_hidden_where_it_stands_bare_in_the_same_request() -> None:
    # Arrange
    random_value = "Zx81kQ0pLw93mN2vB7cR" * 2
    code = f'CASES = [(\'aws_secret_access_key = "{random_value}"\', "{random_value}")]\n'
    state = {"slice": {"file": "tests/test_x.py", "code": code}}

    # Act
    masked_state, questions, masked = mask_request(state, {}, SecretMasker())

    # Assert
    assert random_value not in masked_state["slice"]["code"]
    refuse_if_secret(masked_state, questions, SecretScanner(), masked)


@pytest.mark.parametrize(
    ("state", "rest"),
    [
        (
            {
                "compose": {
                    "file": "docker-compose.yml",
                    "code": (
                        "    environment:\n      POSTGRES_PASSWORD: Zq7wPx@Lm4nRt9vK\n"
                        "      DATABASE_URL: postgres://app:Zq7wPx@Lm4nRt9vK@db/app\n"
                    ),
                }
            },
            "Lm4nRt9vK",
        ),
        (
            {
                "keyed": {"file": "app/fixtures.py", "code": 'password = "Ka9#vQ2mLx7pRt4w"\n'},
                "conf": {"file": "config.yml", "code": "db_pass: Ka9#vQ2mLx7pRt4w\n"},
            },
            "vQ2mLx7pRt4w",
        ),
    ],
)
def test_a_known_value_a_rule_would_cut_short_is_hidden_whole(state: dict, rest: str) -> None:
    """A rule can end a value early: a URL password at its first ``@``, a config value at ``#``. The rest
    of a value the request already knows is hidden where it follows the mask, or it would be sent."""
    # Act
    masked_state, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert rest not in json.dumps(masked_state)


def test_a_quoted_shell_value_is_hidden_where_it_stands_unquoted() -> None:
    # Arrange
    state = {
        "env": {"file": "deploy.sh", "code": 'export DB_PASS="pa55word99xq"\n'},
        "notes": {"file": "docs/restore.md", "code": "psql -h db -W pa55word99xq\n"},
    }

    # Act
    masked_state, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert masked_state["notes"]["code"] == "psql -h db -W [MASKED]\n"


def test_a_token_holding_a_shorter_known_value_is_hidden_whole() -> None:
    # Arrange
    state = {
        "keyed": {"file": "app/fixtures.py", "code": 'password = "Lm4nRt9vKq2w"\n'},
        "ci": {"file": "ci.sh", "code": "curl -H 'x' ghp_Lm4nRt9vKq2wAbCdEfGhIjKlMnOpQrStUvWx12\n"},
    }

    # Act
    masked_state, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert masked_state["ci"]["code"] == "curl -H 'x' [MASKED]\n"


class _OwnTokenMasker(SecretMasker):
    """A masker with a rule of its own that JVN's rules do not have, and that lists only JVN's values."""

    def mask(self, text: str, path: str | None = None) -> str:
        return super().mask(re.sub(r"\btok_[A-Za-z0-9]{20,}", MASK, text), path)


class _OwnEmailMasker(SecretMasker):
    """A masker with an email rule of its own that writes its own token and names it."""

    token_pattern = re.compile(r"\[MASKED\]|\[EMAIL\]")

    def mask(self, text: str, path: str | None = None) -> str:
        return super().mask(re.sub(r"[A-Za-z0-9.]+@[a-z]+\.[a-z]+", "[EMAIL]", text), path)


def test_the_start_of_a_known_value_a_rule_begins_late_is_hidden_beside_the_maskers_own_token() -> None:
    """The email rule starts after the ``#`` inside a known password, so the password's start stands
    before the masker's own token; it is hidden there."""
    # Arrange
    state = {
        "keyed": {"file": "app/fixtures.py", "code": 'password = "Zq7w#Px9@corp.example"\n'},
        "notes": {"file": "docs/login.md", "code": "log in with Zq7w#Px9@corp.example today\n"},
    }

    # Act
    masked_state, questions, values = mask_request(state, {}, _OwnEmailMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert masked_state["notes"]["code"] == "log in with [MASKED] today\n"


def test_a_known_value_inside_a_secret_only_the_masker_finds_leaves_that_secret_whole() -> None:
    """A known value can stand inside a longer secret that only the caller's masker recognizes. Hiding
    the known value first would break that secret's shape and send the rest of it."""
    # Arrange
    state = {
        "keyed": {"file": "app/fixtures.py", "code": 'password = "Lm4nRt9vKq2w"\n'},
        "ci": {"file": "ci.sh", "code": "publish --auth tok_Lm4nRt9vKq2wAbCdEfGh1234\n"},
    }

    # Act
    masked_state, questions, values = mask_request(state, {}, _OwnTokenMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert "AbCdEfGh1234" not in json.dumps(masked_state)


@pytest.mark.parametrize(
    "path", ["tests/test_judgments.py", "tests/test_round.py", "tests/test_secret_masking.py"]
)
def test_the_final_scan_finds_nothing_in_masked_repository_code(path: str) -> None:
    # Arrange
    text = (Path(__file__).parents[1] / path).read_text()

    # Act
    masked = SecretMasker().mask(text, path)

    # Assert
    assert SecretScanner().findings(masked, path) == []


def test_an_unterminated_value_inside_another_string_ends_where_that_string_closes() -> None:
    # Act
    masked = SecretMasker().mask(
        """('DB_PASSWORD="unterminated secret', "DB_PASSWORD=[REDACTED]"),""", "tests/x.py"
    )

    # Assert
    assert masked == """('DB_PASSWORD="[MASKED]', "DB_PASSWORD=[REDACTED]"),"""


def _candidate_request(signature: str, preview: str) -> dict:
    return {"candidates": [{"signature": signature, "preview": preview, "relationship": "calls"}]}


def test_a_candidate_from_a_code_file_keeps_its_code() -> None:
    # Arrange
    line = "createApiKey: (input) => serverClient.apiKeys.create.mutate(input),"
    state = _candidate_request(f"apps/web/src/app/api/settings/api-keys/route.ts:28 `{line}`", f"  {line}\n")

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert masked_state == state


def test_a_candidate_from_a_config_file_is_read_as_config() -> None:
    # Arrange
    state = _candidate_request(
        "deploy/docker-compose.yml:3-12 line 5 `POSTGRES_PASSWORD: example` (mentions)",
        "environment:\n  POSTGRES_PASSWORD: example\n",
    )

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert "example" not in json.dumps(masked_state)


def test_a_candidate_whose_location_does_not_parse_is_read_as_config() -> None:
    # Arrange
    state = _candidate_request("form.ts, line 3: `token: abc123`", "token: abc123\n")

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert "abc123" not in json.dumps(masked_state)


def _check_request(code: str, path: str) -> tuple[dict, dict, frozenset]:
    check = Check(
        "contains_target",
        "Is `slice.code` the admin check that `target.description` describes?",
        Criterion("it is"),
        Criterion("it is not"),
    )
    state = {"target": {"description": "the admin check"}, "slice": {"file": path, "code": code}}
    return mask_request(state, {check.question_id: check.to_question()}, SecretMasker())


def test_a_short_value_equal_to_a_request_key_does_not_refuse_the_request() -> None:
    # Arrange
    state, questions, masked = _check_request('const auth = { useToken: "false" };\n', "src/auth.ts")

    # Act
    refuse_if_secret(state, questions, SecretScanner(), masked)

    # Assert
    assert state["slice"]["code"] == 'const auth = { useToken: "[MASKED]" };\n'


def test_question_wording_and_the_target_keep_their_words_while_the_code_hides_the_value() -> None:
    # Arrange
    state, questions, masked = _check_request('password = "admin"\n', "tests/fixtures.py")

    # Act
    refuse_if_secret(state, questions, SecretScanner(), masked)

    # Assert
    assert state["target"]["description"] == "the admin check"
    assert "the admin check" in json.dumps(questions)
    assert "admin" not in state["slice"]["code"]


@pytest.mark.parametrize(
    "code",
    [
        'docker login --password-stdin < token.txt\nif [ "$a" < "$b" ]; then echo x; fi\n',
        "psql --password $PGPASS -h db\necho $PGPASS\n",
        "deploy --token | tee log.txt\n",
    ],
)
def test_a_flag_without_a_value_or_with_a_reference_keeps_the_line(code: str) -> None:
    # Act
    masked_state, _, _ = mask_request({"slice": {"file": "scripts/run.sh", "code": code}}, {}, SecretMasker())

    # Assert
    assert masked_state["slice"]["code"] == code


def test_a_masked_value_without_letters_or_digits_is_not_copied() -> None:
    # Arrange
    state = {
        "slice": {"file": "app/a.py", "code": 'password = "<>"\n'},
        "other": {"file": "app/b.ts", "code": "if (a <> b) { return a < b; }\n"},
    }

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert masked_state["other"]["code"] == "if (a <> b) { return a < b; }\n"


@pytest.mark.parametrize(("value", "copied"), [("1234", False), ("12345", True)])
def test_a_masked_number_is_copied_only_from_five_characters(value: str, copied: bool) -> None:
    # Arrange
    state = {
        "slice": {"file": "app/settings.ts", "code": f'API_KEY_VERSION: "{value}",\n'},
        "other": {"file": "app/math.ts", "code": f"const limit = {value};\n"},
    }

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert (value not in masked_state["other"]["code"]) is copied


@pytest.mark.parametrize(
    ("value", "text", "hidden"),
    [
        ("hunter2hunter2", "xhunter2hunter2x", True),
        ("hunter2", 'connect(user, "hunter2")', True),
        ("hunter2", "hunter2x and $hunter2", False),
    ],
)
def test_a_copy_pattern_finds_a_long_value_anywhere_and_a_short_one_as_a_whole_word(
    value: str, text: str, hidden: bool
) -> None:
    # Act
    found = copy_pattern(value).search(text)

    # Assert
    assert bool(found) is hidden


@pytest.mark.parametrize(
    ("line", "file", "copied"),
    [
        ('RUNS_REST_TOKEN_HEADER = "x-heedvane-runs-rest-token"\n', "enginepy/hub/auth.py", "token"),
        ('RUNS_REST_TOKEN_HEADER = "x-heedvane-runs-rest-token"\n', "enginepy/hub/auth.py", "heedvane"),
        ('TOKEN_STORE = "/var/lib/app/tokens"\n', "app/settings.py", "tokens"),
        ('TOKEN_ENDPOINT = "https://auth.example.com/oauth/token"\n', "app/settings.py", "oauth"),
    ],
)
def test_a_name_holding_a_copy_of_a_masked_value_stays_a_name_and_the_request_is_sent(
    line: str, file: str, copied: str
) -> None:
    # Arrange
    state = {
        "named": {"file": file, "code": line},
        "keyed": {"file": "app/fixtures.py", "code": f'password = "{copied}"\n'},
    }

    # Act
    masked_state, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert masked_state["named"]["code"] == copy_pattern(copied).sub(MASK, line)


def test_a_value_that_a_copy_turns_into_a_secret_is_masked_again_so_the_request_is_sent() -> None:
    # Arrange
    state = {
        "fixture": {"file": "src/session.test.ts", "code": '  sessionToken: "session-token",\n'},
        "keyed": {"file": "app/fixtures.py", "code": 'password = "session"\n'},
    }

    # Act
    masked_state, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert masked_state["fixture"]["code"] == '  sessionToken: "[MASKED]",\n'


@pytest.mark.parametrize(
    "line",
    [
        '    api_token: str = field(default="admin")\n',
        '    secret: process.env.AUTH_SECRET, role: "admin",\n',
    ],
)
def test_a_code_line_a_copy_changed_is_masked_again_as_code(line: str) -> None:
    # Arrange
    state = {
        "code": {"file": "app/settings.py", "code": line},
        "keyed": {"file": "app/fixtures.py", "code": 'password = "admin"\n'},
    }

    # Act
    masked_state, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert masked_state["code"]["code"] == line.replace('"admin"', '"[MASKED]"')


@pytest.mark.parametrize(
    "point",
    [
        {"target": {"description": "where the shared include list is left empty"}},
        {"targets": {"p0": "where the shared include list is left empty"}},
        {"workflow": {"question": "where the shared include list is left empty"}},
    ],
    ids=["target", "targets", "workflow"],
)
def test_the_point_keeps_a_word_a_code_items_secret_holds(point: dict) -> None:
    # Arrange: a test fixture gives secret-named variables the point's words as values
    item = {"file": "tests/test_paths.py", "code": 'API_TOKEN = "include"\nDB_PASSWORD = "shared"\n'}
    state = {**point, "slice": item}

    # Act
    masked_state, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert masked_state[next(iter(point))] == next(iter(point.values()))
    assert masked_state["slice"]["code"] == 'API_TOKEN = "[MASKED]"\nDB_PASSWORD = "[MASKED]"\n'


def test_the_point_masks_a_secret_the_scanner_finds_in_it() -> None:
    # Arrange
    token = "ghp_" + "Q7rT2mX9vL4kP8wZ3nB6cF1hJ5dS0aGyE2uI"
    state = {
        "target": {"description": f"where the token {token} is sent"},
        "slice": {"file": "a.py", "code": "x = 1\n"},
    }

    # Act
    masked_state, questions, values = mask_request(state, {}, SecretMasker())
    refuse_if_secret(masked_state, questions, SecretScanner(), values)

    # Assert
    assert masked_state["target"]["description"] == "where the token [MASKED] is sent"


def test_a_code_item_hides_a_copy_of_a_secret_the_point_holds() -> None:
    # Arrange
    token = "ghp_" + "Q7rT2mX9vL4kP8wZ3nB6cF1hJ5dS0aGyE2uI"
    state = {
        "target": {"description": f"where {token} is sent"},
        "slice": {"file": "a.py", "code": f"send({token!r})\n"},
    }

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert masked_state["slice"]["code"] == "send('[MASKED]')\n"


def test_a_code_item_under_a_nested_target_key_still_hides_copies() -> None:
    # Arrange: only the request's own point is exempt; a "target" inside an item is code
    item = {
        "file": "tests/test_paths.py",
        "code": 'API_TOKEN = "include1"\n',
        "binding": {"target": "include1"},
    }
    state = {"slice": item}

    # Act
    masked_state, _, _ = mask_request(state, {}, SecretMasker())

    # Assert
    assert masked_state["slice"]["binding"]["target"] == "[MASKED]"
