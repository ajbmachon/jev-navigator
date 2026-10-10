"""The route table: named decision-model routes with automatic fallback.

`SYSTEM_ONE_ROUTES` orders named routes (``drex,jev`` makes Drex primary and Jev its
fallback); each route reads `SYSTEM_ONE_<NAME>_ENDPOINT`, `SYSTEM_ONE_<NAME>_API_KEY` and
`SYSTEM_ONE_<NAME>_MODEL`. Only the jev route may use `TYPESAFE_API_KEY` when it has no key of its
own: every other route is another company's or the user's own server, and must never receive the
TypeSafe key, so a route without its own key is refused before any request. A known route name
with `SYSTEM_ONE_<NAME>=1` uses the hosted defaults, so one flag per service is enough when the
defaults apply. The first route is primary; on a failed call the next route is asked, and so on.
The finetuned decider is just another route: `SYSTEM_ONE_ROUTES=decider,jev` with its endpoint, key, model,
`SYSTEM_ONE_DECIDER_INPUT_TOKENS` and `SYSTEM_ONE_DECIDER_CONCURRENCY` set. Every route declares its
input limits, and the routed client packs to the tightest of them, so whichever route answers can
take the request. Concurrency stays per route: each route's client sends at most its own number of
requests at once, so a slow route never throttles the routes after it.

Every route runs the same generic `SystemOneClient` over the official TypeSafe SDK: the SDK
builds, sends and retries the request with exact-byte capture, and the response is decoded
by jev-navigator's parser. A known route whose server is stricter than Jev names its wire dialect:
Drex's (`drex_wire`) sends criteria as text and breaks base64 data URLs; every other route sends
the request exactly as the judge built it. `TypeSafeJevClient` stays the Jev-specific client (its defaults,
cancellation and Jev model semantics); routes never use it for non-Jev models. The chain
from `environment.py` fills every file-provided variable before routes resolve, so `.env`
configures routes exactly like the real environment does.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from ..environment import ROUTES_ENV, route_names
from ..judgments.answers import JevResponse, response_from_raw
from ..judgments.client import (
    JEV_INPUT_LIMITS,
    LATEST_JEV,
    InputBudgetExceededError,
    InputLimits,
    Wire,
    input_budget_error,
)
from ..judgments.journal import AttemptJournalCallbackError, RawResponse
from .drex_wire import drex_wire

if TYPE_CHECKING:
    from .typesafe import TypeSafeJevClient

JEV_ROUTE = "jev"
TYPESAFE_KEY_SETTING = "TYPESAFE_API_KEY"

DREX_INPUT_LIMITS = InputLimits.from_tokens(8_192)
"""Drex's measured limit: 8,192 tokens for the state plus the longest question. No whole-body bound
is measured, so its own refusal stays authoritative. Historical measurement provenance is retained
in ``measurements/runtime-limits/REPORT.md``."""

DREX_CONCURRENCY = 2
"""Requests Drex admits in flight at once; measurements observed HTTP 429 on the third. Historical
measurement provenance is retained in ``measurements/runtime-limits/REPORT.md``."""

JEV_CONCURRENCY = 32
"""Requests sent to Jev at once; measurements found no 429 up to 128 and flat latency to 32, so 32
is a latency choice, not a refusal bound. Historical measurement provenance is retained in
``measurements/runtime-limits/REPORT.md``."""


@dataclass(frozen=True)
class KnownRoute:
    """A hosted route's endpoint, model, measured input limits and concurrency, and the wire dialect
    its server needs when it is stricter than Jev's (None sends the request as built)."""

    endpoint: str
    model: str
    input_limits: InputLimits
    max_concurrency: int
    wire: Wire | None = None


# Known hosted shorthand routes: `SYSTEM_ONE_<NAME>=1` selects endpoint+model without
# spelling them out. Keys come from `SYSTEM_ONE_<NAME>_API_KEY`; only jev falls back to
# `TYPESAFE_API_KEY`.
KNOWN_ROUTES: dict[str, KnownRoute] = {
    "jev": KnownRoute("https://api.typesafe.ai", LATEST_JEV, JEV_INPUT_LIMITS, JEV_CONCURRENCY),
    "drex": KnownRoute("https://drex.nace.ai", "drex-latest", DREX_INPUT_LIMITS, DREX_CONCURRENCY, drex_wire),
}


@dataclass(frozen=True)
class Route:
    """One decision-model route: a name plus the client that serves it."""

    name: str
    client: SystemOneClient


def system_one_client(environment: Mapping[str, str]) -> TypeSafeJevClient | RoutedJevClient:
    """The client every CLI command judges with: the route table when `SYSTEM_ONE_ROUTES` names
    routes, otherwise the default Jev client. Call it after `.env` has filled ``environment``."""
    routes = routes_from_env(environment)
    if routes:
        return RoutedJevClient(routes)
    from .typesafe import TypeSafeJevClient

    return TypeSafeJevClient(model=None)


def routes_from_env(environment: Mapping[str, str], transport=None) -> tuple[Route, ...]:
    """Resolve the route table from the environment (after `.env` chain loading).

    `SYSTEM_ONE_ROUTES=drex,jev` builds one client per name. A known name with
    `SYSTEM_ONE_<NAME>=1` and no explicit settings uses the hosted shorthand; a known name always
    sends in its server's wire dialect, whichever endpoint it is given. Unknown names
    require endpoint and model; every route resolves its key eagerly (per-route
    `SYSTEM_ONE_<NAME>_API_KEY`, and for jev alone `TYPESAFE_API_KEY`), so a misconfigured route
    fails at resolution instead of mid-run.
    """
    return tuple(_route(environment, name, transport) for name in route_names(environment))


def _route(environment: Mapping[str, str], name: str, transport=None) -> Route:
    upper = name.upper()

    def setting(suffix: str) -> str:
        return str(environment.get(f"SYSTEM_ONE_{upper}_{suffix}", "")).strip()

    endpoint = setting("ENDPOINT")
    model = setting("MODEL")
    known = KNOWN_ROUTES.get(name)
    if known is not None and environment.get(f"SYSTEM_ONE_{upper}", "").strip():
        endpoint = endpoint or known.endpoint
        model = model or known.model

    if not endpoint or not model:
        raise ValueError(
            f"route {name!r} is incomplete: set {ROUTES_ENV} names it, but it needs "
            f"SYSTEM_ONE_{upper}_ENDPOINT and SYSTEM_ONE_{upper}_MODEL"
            + (f" (or SYSTEM_ONE_{upper}=1 for the hosted {name} defaults)" if name in KNOWN_ROUTES else "")
        )

    input_limits = _input_limits(environment, name, known)
    max_concurrency = _concurrency(environment, name, known)
    client = SystemOneClient(
        model=model,
        api_key=_route_key(environment, name),
        base_url=endpoint,
        transport=transport,
        input_limits=input_limits,
        max_concurrency=max_concurrency,
        wire=known.wire if known is not None else None,
    )
    return Route(name=name, client=client)


def _route_key(environment: Mapping[str, str], name: str) -> str:
    """The route's own `SYSTEM_ONE_<NAME>_API_KEY`; the jev route alone may use `TYPESAFE_API_KEY`."""
    own = f"SYSTEM_ONE_{name.upper()}_API_KEY"
    accepted = (own, TYPESAFE_KEY_SETTING) if name == JEV_ROUTE else (own,)
    for setting in accepted:
        if key := str(environment.get(setting, "")).strip():
            return key
    raise ValueError(f"route {name!r} has no key: set {' or '.join(accepted)}")


def _input_limits(environment: Mapping[str, str], name: str, known: KnownRoute | None) -> InputLimits:
    tokens = _route_number(environment, name, "INPUT_TOKENS", required=known is None)
    return known.input_limits if tokens is None else InputLimits.from_tokens(tokens)


def _concurrency(environment: Mapping[str, str], name: str, known: KnownRoute | None) -> int:
    slots = _route_number(environment, name, "CONCURRENCY", required=known is None)
    return known.max_concurrency if slots is None else slots


_ROUTE_NUMBERS = {
    "INPUT_TOKENS": "the tokens it accepts for the state plus the longest question",
    "CONCURRENCY": "how many requests it admits in flight at once",
}


def _route_number(environment: Mapping[str, str], name: str, suffix: str, *, required: bool) -> int | None:
    """`SYSTEM_ONE_<NAME>_<SUFFIX>` as a positive whole number, or None when unset; a custom route
    must set it, since packing and sending must know what the route accepts."""
    setting = f"SYSTEM_ONE_{name.upper()}_{suffix}"
    raw = str(environment.get(setting, "")).strip()
    if not raw:
        if required:
            raise ValueError(f"route {name!r} needs {setting}: {_ROUTE_NUMBERS[suffix]}")
        return None
    if not raw.isdigit() or int(raw) == 0:
        raise ValueError(f"{setting} must be a positive whole number, got {raw!r}")
    return int(raw)


class SystemOneClient:
    """A generic System-One decision client over the official TypeSafe SDK: any endpoint
    speaking the `/v1/systemone` wire (Jev, Drex, a finetuned decider).

    The SDK client owns configuration and auth; its own request builder prepares the request
    and its transport sends it with exact-byte capture. A ``wire`` dialect, when the route has
    one, rewrites the request just before it is sent (Drex's sends criteria as text), so the
    request the judge hashed and stored is unchanged and the captured sent body is what the
    server got. The client's ``input_limits`` measure the request in that same form, so the
    judge splits a request that would only overflow once rewritten. The response is decoded by
    jev-navigator's parser from the exact bytes instead of the SDK's strict response schemas
    (whose score `legend` model rejects Drex's echo shape). Every SDK-internal access lives in
    `_send_raw`, so an SDK version bump is a one-function fix.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str,
        base_url: str | None = None,
        transport=None,
        input_limits: InputLimits,
        max_concurrency: int,
        wire: Wire | None = None,
    ) -> None:
        import httpx2
        from typesafe_sdk import TypeSafeClient

        from .typesafe import CapturingTransport

        self._capture = CapturingTransport(transport or httpx2.HTTPTransport())
        self._sdk = TypeSafeClient(
            model=model or "system-one",
            api_key=api_key,
            base_url=base_url or None,
            transport=self._capture,
        )
        self.model = self._sdk._config.default_model  # noqa: SLF001 - the config is the env contract
        self.input_limits = input_limits if wire is None else replace(input_limits, wire=wire)
        self._slots = threading.BoundedSemaphore(max_concurrency)
        self._wire = wire

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        return self.parse(self.send(state, questions))

    def _send_raw(self, state: Mapping, questions: Mapping) -> None:
        """Prepare with the SDK's own endpoint builder and send through its retry/transport.

        The lenient response type (extra-allow, no required fields) passes the SDK's response
        validation for any System-One answer shape; the captured exact bytes are the truth the
        journal and `parse` read. Returns nothing: the capture and any error carry the result.
        """
        from pydantic import ConfigDict
        from typesafe_sdk._core.endpoints import prepare_system_one
        from typesafe_sdk._core.schemas.base import Response
        from typesafe_sdk._core.transport import send as sdk_send

        class LenientResponse(Response):
            model_config = ConfigDict(extra="allow")

            @classmethod
            def _decode(cls, response):
                return cls.model_validate_json(response.content)

        if self._wire is not None:
            state, questions = self._wire(state, questions)
        request = prepare_system_one(
            self._sdk._config, dict(state), dict(questions), None, None, None, None, LenientResponse
        )  # noqa: SLF001
        sdk_send(self._sdk._http_client, self._sdk._retry, request)  # noqa: SLF001

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        return self.send_with_attempts(state, questions)

    def send_with_attempts(self, state: Mapping, questions: Mapping, on_attempt=None) -> RawResponse:
        """The captured response, with a provider input-budget refusal translated to the typed
        ``InputBudgetExceededError`` so the batching owner can split the batch. Only the send holds
        one of the route's concurrency slots, so a request waiting for a slot has not started and
        cannot time out."""
        with self._capture.collecting(on_attempt) as collection:
            try:
                with self._slots:
                    self._send_raw(state, questions)
            except Exception as error:
                typed = input_budget_error(error)
                if typed is not None:
                    raise typed from error
                raise
            captured = collection.attempts[-1].response if collection.attempts else None
            if captured is None:
                raise RuntimeError("the SDK transport captured no response")
            return RawResponse(
                captured.body,
                captured.status,
                captured.content_type,
                captured.decoded,
                exact=captured.exact,
                sent_body=captured.sent_body,
                attempts=tuple(collection.attempts),
            )

    def close(self) -> None:
        """Release the SDK and its HTTP transport."""
        self._sdk.close()

    def parse(self, raw: RawResponse) -> JevResponse:
        # jev-navigator's parser, not the SDK's strict response schemas: the SDK builds and
        # sends the request, but Drex and the finetuned models echo score legends in shapes
        # the SDK's models reject and the library accepts.
        return response_from_raw(raw.json())


class RoutedJevClient:
    """A `JevClient` over the route table: the first route answers; on failure the next is
    asked, and so on. `model` names the primary route's model, so manifests and journals
    record what was *asked for*; `served_model` on each response records what *answered*."""

    def __init__(self, routes: tuple[Route, ...]) -> None:
        if not routes:
            raise ValueError("RoutedJevClient needs at least one route")
        self.routes = routes
        self.model = routes[0].client.model
        self.input_limits = _tightest(route.client.input_limits for route in routes)

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        return self.parse(self.send(state, questions))

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        return self.send_with_attempts(state, questions)

    def send_with_attempts(self, state: Mapping, questions: Mapping, on_attempt=None) -> RawResponse:
        """Fail over across routes while retaining every route's HTTP attempts as one call."""
        failures: list[str] = []
        attempts = []
        attempt_no = 0
        for route in self.routes:

            def report(attempt, *, route_name=route.name):
                nonlocal attempt_no
                attempt_no += 1
                recorded = replace(attempt, attempt_no=attempt_no, route=route_name)
                attempts.append(recorded)
                if on_attempt is not None:
                    on_attempt(recorded)

            try:
                raw = route.client.send_with_attempts(state, questions, on_attempt=report)
                route.client.parse(raw)  # Preserve the old rule that an unparsable answer triggers failover.
                return replace(raw, attempts=tuple(attempts))
            except AttemptJournalCallbackError as error:
                raise error.original_error from error
            except InputBudgetExceededError as error:
                # A size refusal is about this request's input, which failover would resend unchanged.
                raise _refused_by(route, error) from error
            except Exception as error:  # noqa: BLE001 - failover is the point
                failures.append(f"{route.name}: {error}")
        if not failures:
            raise RuntimeError("no route answered")
        raise ConnectionError("every route failed: " + "; ".join(failures))

    def parse(self, raw: RawResponse) -> JevResponse:
        return response_from_raw(raw.json())

    def close(self) -> None:
        """Release every route's client."""
        for route in self.routes:
            route.client.close()


def _tightest(limits) -> InputLimits:
    tightest, *others = limits
    for other in others:
        tightest = tightest.tightest(other)
    return tightest


def _refused_by(route: Route, error: InputBudgetExceededError) -> InputBudgetExceededError:
    client = route.client
    return InputBudgetExceededError(str(error), model=client.model, box_chars=client.input_limits.box_chars)
