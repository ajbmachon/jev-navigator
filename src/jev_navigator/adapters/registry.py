"""Every decision-model adapter shipped with this library, by the route name that selects it
(`SYSTEM_ONE_ROUTES=<name>`).

To add one: write `adapters/<name>.py` with a subclass of `SystemOneClient` (a model behind HTTP)
or `local.LocalModelClient` (a model in this process) that sets the contract attributes, list it
below, and give it a test for its dialect. `tests/test_adapters.py` then holds it to the shared
contract. An adapter outside this library is not listed: a route names it with
`SYSTEM_ONE_<NAME>_ADAPTER=module:Class`. See "Add a decision-model adapter" in docs/extending.md.
"""

from __future__ import annotations

from .drex import DrexClient
from .system_one import Adapter
from .typesafe import TypeSafeJevClient

ADAPTERS: dict[str, type[Adapter]] = {adapter.name: adapter for adapter in (TypeSafeJevClient, DrexClient)}
