"""Must-pass capabilities for jev-navigator's moves, on a small Express and Next.js style TypeScript app.

Each test is one break the hard benchmark found on a real repository (named in the test), rebuilt small.
The tests use only the library's public moves: open a start place, list what the moves offer with the
current uncapped defaults, and check the place the true path needs is among the kept offers.

Copied in from ``jev-navigator-evals/builder/test_express_capabilities.py`` so the benchmark's must-pass
capabilities run in the library's own test command. The lookalike cases in ``test_places.py``,
``test_imports.py`` and ``test_scope_scan.py`` stay: each of those pins one lookup on a two or three file
snippet (one move at a time, ``find_definition``/``callee_edges``/``symbols_in``), while every case here
walks the whole built index - a start place opened with ``starting_places``, then every default move without
a result cap through ``neighbours_and_omissions``.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from git_repos import commit_files

from jev_navigator.directives.places import MOVES, Place, neighbours_and_omissions, starting_places
from jev_navigator.index.code_index import CodeIndex

TSCONFIG = '{"compilerOptions": {"baseUrl": ".", "paths": {"@/*": ["src/*"]}}}'

APP = """\
import express from 'express';
import { authenticate } from '@/middleware/auth';
import { ordersRouter } from '@/routes/orders';

export const app = express();
app.use(authenticate);
app.use('/orders', ordersRouter);
"""

AUTH = """\
export function authenticate(req, res, next) {
  if (!req.headers.authorization) {
    return res.status(401).json({ error: 'unauthorized' });
  }
  return next();
}
"""

ORDERS_ROUTER = """\
import { Router } from 'express';
import { createOrder } from '@/services';

export const ordersRouter = Router();

ordersRouter.post('/', async (req, res) => {
  const order = await createOrder(req.body);
  res.json(order);
});
"""

NEXT_ROUTE = """\
import { createOrder } from '@/services';
import { parseRequest } from '@/lib/request';

export async function POST(request) {
  const { body, error } = await parseRequest(request);
  if (error) {
    return error();
  }
  const token = request.headers.get('authorization');
  const cached = request.headers.get('x-cache');
  const trace = request.headers.get('x-trace');
  const locale = request.headers.get('accept-language');
  const agent = request.headers.get('user-agent');
  const origin = request.headers.get('origin');
  const order = await createOrder({ ...body, token, cached, trace, locale, agent, origin });
  return Response.json(order);
}
"""

REQUEST = """\
export async function parseRequest(request) {
  const body = await request.json();
  const error = body ? undefined : () => new Response(null, { status: 400 });
  return { body, error };
}
"""

CACHE = """\
export const cache = {
  get(key) {
    return undefined;
  },
};
"""

CACHE_TEST = """\
const headers = { get(name) { return name; } };
const failing = { error: () => new Response(null, { status: 400 }) };
"""

SERVICES_INDEX = """\
export * from './orders';
export * from './invoices';
"""

ORDERS_SERVICE = """\
import { db } from '@/db';
import { runQuery, POSTGRES, CACHED } from '@/lib/query';

export async function createOrder(input) {
  const order = normalise(input);
  return db.$transaction(async (tx) => {
    return tx.order.create({ data: order });
  });
}

function normalise(input) {
  return { ...input };
}

export async function findOrder(id) {
  return runQuery({
    [POSTGRES]: () => relationalFind(id),
    [CACHED]: () => cachedFind(id),
  });
}

async function relationalFind(id) {
  return db.order.findUnique({ where: { id } });
}

async function cachedFind(id) {
  return undefined;
}
"""

INVOICES_SERVICE = """\
export async function createInvoice(input) {
  return { ...input };
}
"""

QUERY = """\
export const POSTGRES = 'postgres';
export const CACHED = 'cached';

export async function runQuery(queries) {
  if (process.env.CACHE_URL) {
    return queries[CACHED]();
  }
  return queries[POSTGRES]();
}
"""

DB = """\
export const db = {};
"""

FLOW_ADAPTER = """\
// @flow
import type { StorageAdapter, QueryOptions } from './StorageAdapter';

const toPostgresValue = (value: any, options: ?QueryOptions): any => {
  const fields: Array<string>[] = [];
  return value;
};

export class PostgresAdapter implements StorageAdapter {
  _client: any;

  constructor({ uri }: { uri: string }) {
    this._client = connect(uri);
  }

  async createObject(className: string, object: Object, options: ?QueryOptions): Promise<void> {
    await this._client.none('INSERT INTO $1:name', [className, toPostgresValue(object, options)]);
  }

  find(className: string, query: Object): Promise<Array<Object>> {
    return this._client.any('SELECT * FROM $1:name', [className, query]);
  }
}

function connect(uri: string): any {
  return { uri };
}
"""

ADAPTER_CHOICE = """\
import { MemoryAdapter } from './memory';
import { PostgresAdapter } from './postgres';

export function getAdapter(uri) {
  if (uri.startsWith('memory:')) {
    return new MemoryAdapter();
  }
  return new PostgresAdapter({ uri });
}
"""

MEMORY_ADAPTER = """\
export class MemoryAdapter {
  constructor() {
    this.rows = [];
  }

  createObject(className, object) {
    this.rows.push({ className, object });
  }
}
"""

FILES = {
    "tsconfig.json": TSCONFIG,
    "src/app.ts": APP,
    "src/middleware/auth.ts": AUTH,
    "src/routes/orders.ts": ORDERS_ROUTER,
    "src/app/api/orders/route.ts": NEXT_ROUTE,
    "src/lib/request.ts": REQUEST,
    "src/lib/cache.ts": CACHE,
    "src/lib/cache.test.ts": CACHE_TEST,
    "src/lib/query.ts": QUERY,
    "src/services/index.ts": SERVICES_INDEX,
    "src/services/orders.ts": ORDERS_SERVICE,
    "src/services/invoices.ts": INVOICES_SERVICE,
    "src/db.ts": DB,
    "src/adapters/postgres.js": FLOW_ADAPTER,
    "src/adapters/index.js": ADAPTER_CHOICE,
    "src/adapters/memory.js": MEMORY_ADAPTER,
}


@pytest.fixture
def shop_index(tmp_path: Path) -> CodeIndex:
    """The shop as one real repository: the files are written and committed, then indexed from git."""
    root = tmp_path / "shop"
    commit_files(root, FILES)
    return CodeIndex.from_git(root)


def _kept_offers(index: CodeIndex, file: str, line: int) -> list[Place]:
    start = starting_places(index, [(file, line)])[0]
    kept, _omitted = neighbours_and_omissions(index, start.open())
    return kept


def _offers_function(offers: list[Place], file: str, name_line: int) -> bool:
    return any(place.key.startswith(f"{file}:{name_line}-") for place in offers)


def test_a_registration_line_offers_the_function_it_registers(shop_index: CodeIndex) -> None:
    """documenso D5 and parse-server P5: `app.use(authenticate)` is a module-level line, so the start is a
    window with no name and no call or reference move starts from it."""
    # Act
    offers = _kept_offers(shop_index, "src/app.ts", 6)

    # Assert
    assert _offers_function(offers, "src/middleware/auth.ts", 1)


def test_an_unnamed_route_handler_offers_what_it_calls(shop_index: CodeIndex) -> None:
    """parse-server P1: the route line opens the anonymous arrow `async (req, res) => {...}`, and every
    call move needs a named function, so createOrder is never offered."""
    # Act
    offers = _kept_offers(shop_index, "src/routes/orders.ts", 6)

    # Assert
    assert _offers_function(offers, "src/services/orders.ts", 4)


def test_a_function_imported_through_a_barrel_file_is_a_proven_callee(shop_index: CodeIndex) -> None:
    """umami U1: `import { saveEvent } from '@/queries/sql'`, where index.ts re-exports with `export *`,
    leaves saveEvent as a name-only candidate."""
    # Act
    offers = _kept_offers(shop_index, "src/app/api/orders/route.ts", 4)

    # Assert
    create_order = [place for place in offers if place.key.startswith("src/services/orders.ts:4-")]
    assert create_order
    assert "candidate" not in create_order[0].signature


def test_proven_callees_are_kept_before_name_only_candidates(shop_index: CodeIndex) -> None:
    """umami U1: POST's first eight callee offers were parseRequest and name-only matches for `error` and
    `get` (two of them in test files); the imported saveEvent came 21st of 29 and fell past the cap."""
    # Arrange
    start = starting_places(shop_index, [("src/app/api/orders/route.ts", 4)])[0]

    # Act
    kept, omitted = neighbours_and_omissions(shop_index, start.open(), per_kind=2)

    # Assert
    assert _offers_function(kept, "src/services/orders.ts", 4)
    assert not any(place.key.startswith("src/lib/cache.test.ts") for place in kept)


def test_a_database_call_in_a_transaction_callback_leads_back_to_the_function_passing_it(
    shop_index: CodeIndex,
) -> None:
    """documenso D6: the writes sit in `prisma.$transaction(async (tx) => {...})`; the callback has no name,
    so no caller move leads from the write back towards the route."""
    # Act
    offers = _kept_offers(shop_index, "src/services/orders.ts", 7)

    # Assert
    assert _offers_function(offers, "src/services/orders.ts", 4)


def test_a_flow_typed_class_keeps_its_methods(shop_index: CodeIndex) -> None:
    """parse-server P1, P2, P4 and P7: DatabaseController.js and PostgresStorageAdapter.js carry Flow types
    (`// @flow`, `implements`, typed parameters). The JavaScript grammar loses 31 of 37 and 41 of 47 of
    their methods, the index still counts the files as parsed, and no move can reach createObject."""
    # Act
    methods = {span.name for span in shop_index.functions_in("src/adapters/postgres.js")}

    # Assert
    assert {"constructor", "createObject", "find"} <= methods


def test_a_constructor_call_offers_the_class_it_builds(shop_index: CodeIndex) -> None:
    """parse-server P4: getDatabaseAdapter returns `new PostgresStorageAdapter({...})`; the path to the
    adapter the server uses starts with that constructor call. The plain class here keeps Flow out of it."""
    # Arrange
    get_adapter = starting_places(shop_index, [("src/adapters/index.js", 4)])[0].open()

    # Act
    callees = neighbours_and_omissions(shop_index, get_adapter, moves={"callees": MOVES["callees"]})[0]

    # Assert
    assert any(place.key.startswith("src/adapters/memory.js:") for place in callees)
