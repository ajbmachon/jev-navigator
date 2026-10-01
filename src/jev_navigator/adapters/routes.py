"""The route table: named decision-model routes with automatic fallback, the pattern
analysis-engine proven on its System One seats.

`SYSTEM_ONE_ROUTES` orders named routes (``drex,jev`` makes Drex primary and Jev its
fallback). The first route is primary; on a failed call the next route is asked, and so on.

A question type can have its own chain: `SYSTEM_ONE_ROUTES_CHECK`, `SYSTEM_ONE_ROUTES_PICK` and
`SYSTEM_ONE_ROUTES_RATE` order the routes for `Check`, `Pick` and `Rate` questions, and replace
`SYSTEM_ONE_ROUTES` for that type, so list a type's fallbacks in its own table. A type without a
table uses `SYSTEM_ONE_ROUTES`, or the registered ``jev`` route when that is unset too. A request
that mixes types bound for different chains is split (see `RoutedJevClient`).

Every route resolves the same way, with these settings for a route named ``<NAME>``:

- `SYSTEM_ONE_<NAME>_ADAPTER`: the adapter class as ``module:Class``, for one that lives outside
  this library, such as your own model running in this process. Unset, a name in
  `registry.ADAPTERS` runs that adapter, so `SYSTEM_ONE_ROUTES=drex` with `DREX_API_KEY` is a
  complete setup, and any other name runs the generic `SystemOneClient`, which needs an endpoint
  and a model: the finetuned decider is `SYSTEM_ONE_ROUTES=decider,jev` with those set.
- `SYSTEM_ONE_<NAME>_ENDPOINT` and `SYSTEM_ONE_<NAME>_MODEL`: beat the adapter's defaults.
- `SYSTEM_ONE_<NAME>_API_KEY`: the route's key; else the adapter's own key variable, never
  another service's. Only an adapter that ``needs_key`` refuses to run without one, so a
  self-hosted server can run keyless.
- `SYSTEM_ONE_<NAME>_TIMEOUT` (seconds per attempt) and `SYSTEM_ONE_<NAME>_RETRIES`: beat the
  adapter's defaults, for a server slower or flakier than a hosted API.
- `SYSTEM_ONE_<NAME>_NOUL_YES_AT`, `SYSTEM_ONE_<NAME>_NOUL_NO_AT` and
  `SYSTEM_ONE_<NAME>_CHOICE_MIN_CONFIDENCE`: the bars this route's model needs, for a model that
  runs hotter or colder than the shared thresholds assume. Its answers are mapped onto the shared
  scale, the `JEV_NAVIGATOR_*` thresholds of the same environment (`thresholds.Calibration`),
  before the judge reads them; the journal keeps them as sent.

A route named in several tables is one adapter, so an in-process model loads once. The chain
from `environment.py` fills every file-provided variable before routes resolve, so `.env`
configures routes exactly like the real environment does.
"""

from __future__ import annotations

import importlib
import math
import os
import threading
from collections.abc import Mapping
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import dataclass, replace

from ..judgments.answers import ChoiceAnswer, JevResponse, NoulAnswer, response_from_raw, response_to_raw
from ..judgments.client import InputBudgetExceededError
from ..judgments.journal import RawResponse
from ..judgments.thresholds import ENVIRONMENT_NAMES, Calibration, Thresholds
from .registry import ADAPTERS
from .system_one import CONTRACT_ATTRIBUTES, Adapter, SystemOneClient

ROUTES_ENV = "SYSTEM_ONE_ROUTES"
# The library's name for each wire question type, and the variable that routes it.
QUESTION_KINDS = {"noul": "check", "choice": "pick", "score": "rate"}
QUESTION_ROUTES_ENV = {kind: f"{ROUTES_ENV}_{kind.upper()}" for kind in QUESTION_KINDS.values()}
MAX_SPLIT_WORKERS = 16


@dataclass(frozen=True)
class Route:
    """One decision-model route: a name, the adapter that serves it, and how its answers map onto
    the shared thresholds when its model needs bars of its own."""

    name: str
    client: Adapter
    calibration: Calibration | None = None


@dataclass(frozen=True)
class RouteTable:
    """``default`` answers every question type without a chain of its own in ``by_kind``
    (keyed ``check``, ``pick``, ``rate``)."""

    default: tuple[Route, ...]
    by_kind: Mapping[str, tuple[Route, ...]]


def routes_from_env(
    environment: Mapping[str, str] | None = None,
    transport=None,
) -> tuple[Route, ...]:
    """The routes `SYSTEM_ONE_ROUTES` orders (after `.env` chain loading)."""
    return route_table_from_env(environment, transport).default


def route_table_from_env(
    environment: Mapping[str, str] | None = None,
    transport=None,
) -> RouteTable:
    """Resolve every route table from the environment (after `.env` chain loading).

    Every route resolves its key eagerly, so a misconfigured route fails at resolution instead
    of mid-run. ``transport`` is handed to every adapter that speaks HTTP, for tests.
    """
    environment = os.environ if environment is None else environment
    unknown = sorted(
        key
        for key in environment
        if key.startswith(f"{ROUTES_ENV}_") and key not in QUESTION_ROUTES_ENV.values()
    )
    if unknown:
        raise ValueError(
            f"{unknown[0]} names no question type; the per-type tables are "
            f"{', '.join(QUESTION_ROUTES_ENV.values())}"
        )
    built: dict[str, Route] = {}

    def chain(names: tuple[str, ...]) -> tuple[Route, ...]:
        for name in names:
            if name not in built:
                built[name] = _route(environment, name, transport)
        return tuple(built[name] for name in names)

    by_kind = {
        kind: chain(names)
        for kind, variable in QUESTION_ROUTES_ENV.items()
        if (names := _names(environment.get(variable, "")))
    }
    default = chain(_names(environment.get(ROUTES_ENV, "")))
    if by_kind and not default and len(by_kind) < len(QUESTION_KINDS):
        default = chain(("jev",))
    return RouteTable(default, by_kind)


def covers_every_question(environment: Mapping[str, str]) -> bool:
    """Whether the route tables leave no question type to the unconfigured Jev default."""
    return bool(_names(environment.get(ROUTES_ENV, ""))) or all(
        _names(environment.get(variable, "")) for variable in QUESTION_ROUTES_ENV.values()
    )


def client_from_env(
    environment: Mapping[str, str] | None = None, transport=None
) -> Adapter | RoutedJevClient | None:
    """The client the route tables configure, or None when they name no route. A single route
    runs its adapter directly; more, or a per-type table, run through `RoutedJevClient`."""
    table = route_table_from_env(environment, transport)
    if table.by_kind:
        return RoutedJevClient(table.default, table.by_kind)
    if len(table.default) == 1 and table.default[0].calibration is None:
        return table.default[0].client
    return RoutedJevClient(table.default) if table.default else None


def _names(text: str) -> tuple[str, ...]:
    return tuple(name.strip().lower() for name in str(text).split(",") if name.strip())


def _route(environment: Mapping[str, str], name: str, transport=None) -> Route:
    prefix = f"SYSTEM_ONE_{name.upper()}"

    def setting(key: str) -> str:
        return str(environment.get(key, "")).strip()

    adapter_path = setting(f"{prefix}_ADAPTER")
    adapter = (
        _imported(f"{prefix}_ADAPTER", adapter_path) if adapter_path else ADAPTERS.get(name, SystemOneClient)
    )
    endpoint = setting(f"{prefix}_ENDPOINT")
    model = setting(f"{prefix}_MODEL")
    if adapter is SystemOneClient and not (endpoint and model):
        raise ValueError(
            f"route {name!r} is incomplete: a route table names it, but it needs {prefix}_ENDPOINT "
            f"and {prefix}_MODEL, or {prefix}_ADAPTER (the registered routes, which need none of "
            f"these, are: {', '.join(ADAPTERS)})"
        )
    key_names = [f"{prefix}_API_KEY", *([adapter.api_key_env] if adapter.api_key_env else [])]
    api_key = next((setting(key) for key in key_names if setting(key)), "")
    if adapter.needs_key and not api_key:
        raise ValueError(f"route {name!r} has no API key: set {' or '.join(key_names)}")
    timeout = _seconds(f"{prefix}_TIMEOUT", setting(f"{prefix}_TIMEOUT"))
    max_retries = _count(f"{prefix}_RETRIES", setting(f"{prefix}_RETRIES"))
    try:
        client = adapter(
            model=model or None,
            api_key=api_key or None,
            endpoint=endpoint or None,
            transport=None if adapter.runs_locally else transport,
            timeout=timeout,
            max_retries=max_retries,
        )
    except ValueError as error:
        raise ValueError(f"route {name!r} ({prefix}_*): {error}") from None
    return Route(name=name, client=client, calibration=_calibration(environment, name, prefix, setting))


def _calibration(environment: Mapping[str, str], name: str, prefix: str, setting) -> Calibration | None:
    """The route's own bars against the shared thresholds, or None when it sets none that differ."""
    own = {
        field: bar
        for field in ENVIRONMENT_NAMES
        if (bar := _fraction(f"{prefix}_{field.upper()}", setting(f"{prefix}_{field.upper()}"))) is not None
    }
    if not own:
        return None
    shared = Thresholds.from_env(environment)
    try:
        own_thresholds = shared.updated(own)
        return Calibration(own_thresholds, shared) if own_thresholds != shared else None
    except ValueError as error:
        raise ValueError(f"route {name!r} thresholds: {error}") from None


def _imported(variable: str, path: str) -> type[Adapter]:
    """The adapter class ``module:Class`` names, checked against the contract before it is built."""
    module_name, _, attribute = path.partition(":")
    if not (module_name and attribute):
        raise ValueError(f"{variable} must name the adapter as module:Class, not {path!r}")
    try:
        adapter = getattr(importlib.import_module(module_name), attribute)
    except (ImportError, AttributeError) as error:
        raise ValueError(f"{variable}={path} cannot be loaded: {error}") from error
    missing = [attribute for attribute in CONTRACT_ATTRIBUTES if not hasattr(adapter, attribute)]
    if not isinstance(adapter, type) or missing:
        raise ValueError(
            f"{variable}={path} is not an adapter class"
            + (f": it lacks {', '.join(missing)}" if isinstance(adapter, type) else "")
        )
    return adapter


def _seconds(variable: str, text: str) -> float | None:
    if not text:
        return None
    try:
        seconds = float(text)
    except ValueError:
        seconds = math.nan
    if not (math.isfinite(seconds) and seconds > 0):
        raise ValueError(f"{variable} must be a positive number of seconds, not {text!r}")
    return seconds


def _fraction(variable: str, text: str) -> float | None:
    if not text:
        return None
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{variable} must be a number from 0 to 1, not {text!r}")
    return value


def _count(variable: str, text: str) -> int | None:
    if not text:
        return None
    if not text.isdecimal():
        raise ValueError(f"{variable} must be a whole number, 0 or more, not {text!r}")
    return int(text)


class RoutedJevClient:
    """A `JevClient` over route chains. Each question goes to the chain for its type in
    ``by_kind``, else to ``routes``; in a chain the first route answers and on failure the next
    is asked.

    A request whose questions all belong to one chain is sent whole, and the response keeps the
    answering route's exact bytes. A request that spans chains is split: each chain gets its own
    questions in one call, the calls run at once, and their answers merge into one response. A
    split request is one call to the judge's budget however many services it reaches, and it
    fails if any part fails. The merged response is journaled decoded, and its model names each
    type's model (``check=mine@1,pick=drex-v1.5``) unless one model answered every part.

    ``model`` names what is asked for, each type's primary model in the same form, so a resumed
    search must use the same routing; `served_model` on each response records what answered.

    A route with bars of its own has its answers calibrated onto the shared thresholds before
    they are returned; the response bytes, which the journal keeps, stay as the service sent them.
    ``route_thresholds`` lists those routes' bars, for the manifest and the resume check.
    """

    def __init__(
        self, routes: tuple[Route, ...], by_kind: Mapping[str, tuple[Route, ...]] | None = None
    ) -> None:
        self.routes = routes
        self.by_kind = dict(by_kind or {})
        if not routes and len(self.by_kind) < len(QUESTION_KINDS):
            raise ValueError(
                "RoutedJevClient needs at least one route"
                + (" for the question types without their own" if self.by_kind else "")
            )
        self.model = _named_by_kind(
            {kind: self._chain(kind)[0].client.model for kind in QUESTION_KINDS.values()}
        )
        chains = (routes, *self.by_kind.values())
        self._clients = tuple(
            {id(route.client): route.client for chain in chains for route in chain}.values()
        )
        self.route_thresholds = {
            route.name: route.calibration.own.as_dict()
            for chain in chains
            for route in chain
            if route.calibration is not None
        }
        self._pool: ThreadPoolExecutor | None = None
        self._pool_lock = threading.Lock()

    def ask(self, state: Mapping, questions: Mapping) -> JevResponse:
        return self.parse(self.send(state, questions))

    def send(self, state: Mapping, questions: Mapping) -> RawResponse:
        parts = self._parts(questions)
        if len(parts) < 2:
            kinds, chain, _ = parts[0] if parts else ((), self._chain("check"), {})
            raw, _, used = self._send_to_chain(kinds, chain, state, questions)
            return replace(raw, decoded=response_to_raw(used))
        with self._pool_lock:
            if self._pool is None:
                self._pool = ThreadPoolExecutor(MAX_SPLIT_WORKERS, thread_name_prefix="jev-routes")
            pool = self._pool
        later = [
            pool.submit(self._send_to_chain, kinds, chain, state, part) for kinds, chain, part in parts[1:]
        ]
        kinds, chain, part = parts[0]
        answered = [self._send_to_chain(kinds, chain, state, part), *(future.result() for future in later)]
        pairs = list(zip((kinds for kinds, _, _ in parts), answered, strict=True))
        received = _merged(questions, [(kinds, received) for kinds, (_, received, _) in pairs])
        used = _merged(questions, [(kinds, used) for kinds, (_, _, used) in pairs])
        return replace(RawResponse.from_decoded(received), decoded=used)

    def parse(self, raw: RawResponse) -> JevResponse:
        return response_from_raw(raw.json())

    def cancel(self) -> None:
        for client in self._clients:
            client.cancel()

    def close(self) -> None:
        for client in self._clients:
            client.close()
        if self._pool is not None:
            self._pool.shutdown()

    def _chain(self, kind: str) -> tuple[Route, ...]:
        return self.by_kind.get(kind) or self.routes

    def _parts(self, questions: Mapping) -> list[tuple[tuple[str, ...], tuple[Route, ...], dict]]:
        """The questions grouped by the chain that answers them, with the kinds each group holds."""
        parts: dict[tuple[str, ...], tuple[list[str], tuple[Route, ...], dict]] = {}
        for question_id, question in questions.items():
            kind = QUESTION_KINDS.get(question.get("type"), str(question.get("type")))
            chain = self._chain(kind)
            kinds, _, part = parts.setdefault(tuple(route.name for route in chain), ([], chain, {}))
            part[question_id] = question
            if kind not in kinds:
                kinds.append(kind)
        return [(tuple(kinds), chain, part) for kinds, chain, part in parts.values()]

    def _send_to_chain(
        self, kinds: tuple[str, ...], chain: tuple[Route, ...], state: Mapping, questions: Mapping
    ) -> tuple[RawResponse, JevResponse, JevResponse]:
        """The first answer in ``chain``: as sent, parsed as received, and as the judge will use it."""
        failures: list[str] = []
        for route in chain:
            try:
                raw = route.client.send(state, questions)
                received = route.client.parse(raw)
            except CancelledError:
                raise
            except InputBudgetExceededError:
                # A size refusal is about this request's input, which failover would resend
                # unchanged, and the typed error is the batching owner's signal to split the batch.
                # It passes through untouched, exactly as `SystemOneClient.send` raises it.
                raise
            except Exception as error:  # noqa: BLE001 - failover is the point
                failures.append(f"{route.name}: {error}")
                continue
            return raw, received, _calibrated(received, route.calibration)
        asked = f" for {' and '.join(kinds)} questions" if self.by_kind and kinds else ""
        raise ConnectionError(f"every route{asked} failed: " + "; ".join(failures))


def _calibrated(response: JevResponse, calibration: Calibration | None) -> JevResponse:
    if calibration is None:
        return response
    answers = {}
    for question_id, answer in response.answers.items():
        if isinstance(answer, NoulAnswer):
            answer = replace(answer, probability=calibration.noul_probability(answer.probability))
        elif isinstance(answer, ChoiceAnswer):
            answer = replace(answer, confidence=calibration.choice_confidence(answer.confidence))
        answers[question_id] = answer
    return replace(response, answers=answers)


def _merged(questions: Mapping, parts: list[tuple[tuple[str, ...], JevResponse]]) -> dict:
    answers = {
        question_id: answer.to_json()
        for _, response in parts
        for question_id, answer in response.answers.items()
    }
    models = {kind: response.model for kinds, response in parts for kind in kinds}
    return {
        "model": _named_by_kind(models),
        "usage": {"input_tokens": sum(response.input_tokens for _, response in parts)},
        "answers": {question_id: answers[question_id] for question_id in questions if question_id in answers},
    }


def _named_by_kind(models: Mapping[str, str]) -> str:
    """One model's name when it serves every kind, else each kind's, in `QUESTION_KINDS` order."""
    if len(set(models.values())) == 1:
        return next(iter(models.values()))
    return ",".join(f"{kind}={models[kind]}" for kind in QUESTION_KINDS.values() if kind in models)
