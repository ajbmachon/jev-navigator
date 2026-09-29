"""The route table: named decision-model routes with automatic fallback, the pattern
analysis-engine proven on its System One seats.

`SYSTEM_ONE_ROUTES` orders named routes (``drex,jev`` makes Drex primary and Jev its
fallback); each route reads `SYSTEM_ONE_<NAME>_ENDPOINT`, `SYSTEM_ONE_<NAME>_API_KEY` and
`SYSTEM_ONE_<NAME>_MODEL`. A known route name with `SYSTEM_ONE_<NAME>=1` uses the hosted
defaults, so one flag per service is enough when the defaults apply. The first route is
primary; on a failed call the next route is asked, and so on. The finetuned decider is just
another route: `SYSTEM_ONE_ROUTES=decider,jev` with its endpoint, key and model set.

Every route runs the same generic `SystemOneClient` over the official TypeSafe SDK: the SDK
builds, sends and retries the request with exact-byte capture, and the response is decoded
by jev-navigator's parser. `TypeSafeJevClient` stays the Jev-specific client (its defaults,
cancellation and Jev model semantics); routes never use it for non-Jev models. The chain
from `environment.py` fills every file-provided variable before routes resolve, so `.env`
configures routes exactly like the real environment does.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

from ..judgments.answers import JevResponse, response_from_raw
from ..judgments.client import LATEST_JEV
from ..judgments.journal import RawResponse

ROUTES_ENV = "SYSTEM_ONE_ROUTES"

# Known hosted shorthand routes: `SYSTEM_ONE_<NAME>=1` selects endpoint+model without
# spelling them out. Keys always come from `SYSTEM_ONE_<NAME>_API_KEY`, falling back to
# `TYPESAFE_API_KEY` for single-route setups.
KNOWN_ROUTES: dict[str, tuple[str, str]] = {
    "jev": ("https://api.typesafe.ai", LATEST_JEV),
    "drex": ("https://drex.nace.ai", "drex-latest"),
}


@dataclass(frozen=True)
class Route:
    """One decision-model route: a name plus the client that serves it."""

    name: str
    client: SystemOneClient


def routes_from_env(
    environment: Mapping[str, str] | None = None,
    transport=None,
) -> tuple[Route, ...]:
    """Resolve the route table from the environment (after `.env` chain loading).

    `SYSTEM_ONE_ROUTES=drex,jev` builds one client per name. A known name with
    `SYSTEM_ONE_<NAME>=1` and no explicit settings uses the hosted shorthand. Unknown names
    require endpoint and model; every route resolves its key eagerly (per-route
    `SYSTEM_ONE_<NAME>_API_KEY`, else `TYPESAFE_API_KEY`), so a misconfigured route fails at
    resolution instead of mid-run.
    """
    environment = os.environ if environment is None else environment
    names = tuple(name.strip().lower() for name in environment.get(ROUTES_ENV, "").split(",") if name.strip())
    if not names:
        return ()
    return tuple(_route(environment, name, transport) for name in names)


def _route(environment: Mapping[str, str], name: str, transport=None) -> Route:
    upper = name.upper()

    def setting(suffix: str) -> str:
        return str(environment.get(f"SYSTEM_ONE_{upper}_{suffix}", "")).strip()

    endpoint = setting("ENDPOINT")
    model = setting("MODEL")
    api_key = setting("API_KEY") or str(environment.get("TYPESAFE_API_KEY", "")).strip() or None
    if name in KNOWN_ROUTES and environment.get(f"SYSTEM_ONE_{upper}", "").strip():
        default_endpoint, default_model = KNOWN_ROUTES[name]
        endpoint = endpoint or default_endpoint
        model = model or default_model

    if not endpoint or not model:
        raise ValueError(
            f"route {name!r} is incomplete: set {ROUTES_ENV} names it, but it needs "
            f"SYSTEM_ONE_{upper}_ENDPOINT and SYSTEM_ONE_{upper}_MODEL"
            + (f" (or SYSTEM_ONE_{upper}=1 for the hosted {name} defaults)" if name in KNOWN_ROUTES else "")
        )

    return Route(
        name=name,
        client=SystemOneClient(model=model, api_key=api_key, base_url=endpoint, transport=transport),
    )


class SystemOneClient:
    """A generic System-One decision client over the official TypeSafe SDK: any endpoint
    speaking the `/v1/systemone` wire (Jev, Drex, a finetuned decider).

    The SDK client owns configuration and auth; its own request builder prepares the request
    and its transport sends it with exact-byte capture. Criteria are flattened to strings on
    the wire (the strictest dialect — Drex's — requires it) and the response is decoded by
    jev-navigator's parser from the exact bytes instead of the SDK's strict response schemas
    (whose score `legend` model rejects Drex's echo shape). Every SDK-internal access lives
    in `_send_raw`, so an SDK version bump is a one-function fix.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        transport=None,
    ) -> None:
        import httpx2
        from typesafe_sdk import TypeSafeClient

        from .typesafe import CapturingTransport

        self._capture = CapturingTransport(transport or httpx2.HTTPTransport())
        self._sdk = TypeSafeClient(
            model=model or "system-one",
            api_key=api_key or None,
            base_url=base_url or None,
            transport=self._capture,
        )
        self.model = self._sdk._config.default_model  # noqa: SLF001 - the config is the env contract

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

        request = prepare_system_one(
            self._sdk._config, dict(state), dict(questions), None, None, None, None, LenientResponse
        )  # noqa: SLF001
        sdk_send(self._sdk._http_client, self._sdk._retry, request)  # noqa: SLF001

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        self._send_raw(state, questions)
        captured = self._capture.take()
        if captured is None:
            raise RuntimeError("the SDK transport captured no response")
        return captured

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

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        failures: list[str] = []
        for route in self.routes:
            try:
                return route.client.ask(state, questions)
            except Exception as error:  # noqa: BLE001 - failover is the point
                failures.append(f"{route.name}: {error}")
        if not failures:
            raise RuntimeError("no route answered")
        raise ConnectionError("every route failed: " + "; ".join(failures))
