"""Jev judgments as building blocks: yes/no checks over supplied items, picks over code-built
options, and one function-calling decision per step.

Every request is masked, scanned, hashed and looked up in the answer store before any call, and
every fresh answer is stored with the thresholds that were in force.
"""

from __future__ import annotations

import asyncio
import base64
import copy
import inspect
import json
import logging
import threading
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field

from .answers import JevResponse, NoulAnswer, TokenTotal, response_to_raw
from .client import (
    JEV_INPUT_BOX_CHARS,
    MAX_REQUEST_CHARS,
    AsyncJevClient,
    InputBudgetExceededError,
    JevClient,
)
from .journal import AttemptJournalCallbackError, Journal, JournalRequest, RawResponse
from .questions import (
    Check,
    Pick,
    Rate,
    content_hash,
    item_path,
    request_body,
    request_sha256,
    serialized_chars,
)
from .secrets import (
    Masker,
    Scanner,
    SecretMasker,
    SecretScanner,
    mask_everywhere,
    mask_request,
    masked_values,
    refuse_if_secret,
    safe_options,
)
from .store import AnswerRecord, AnswerStore
from .thresholds import NoulVerdict, Thresholds

logger = logging.getLogger(__name__)

MAX_STATE_CHARS = 60_000
CODE_FIELD = "code"
ROUTE_QUESTION = "route"
_DEFAULT_MASKER = SecretMasker()
_DEFAULT_SCANNER = SecretScanner()


class CallCapReachedError(RuntimeError):
    """A call would exceed the ``max_calls`` cap of this judge or of a judge it was scoped from."""


def request_exceeds_input_budget(state: Mapping, questions: Mapping) -> bool:
    """Whether a request is outside the character boxes the batching owner sends within.

    The state plus the longest single question must fit ``JEV_INPUT_BOX_CHARS`` (Jev's documented
    32,000 tokens), the limit the Engine measured on 27.09.2026 (32,883 tokens pass, about 33,200
    are refused). The whole body must fit ``MAX_REQUEST_CHARS``, since packing bounds a batch's
    state but not the questions asked of it. Both boxes measure the serialization the body uses.
    This is a preflight; a provider's typed refusal remains authoritative.
    """
    longest_question = max((serialized_chars(question) for question in questions.values()), default=0)
    if serialized_chars(state) + longest_question > JEV_INPUT_BOX_CHARS:
        return True
    return serialized_chars({"state": state, "questions": questions}) > MAX_REQUEST_CHARS


@dataclass(frozen=True)
class CheckResult:
    """``probability`` is Jev's raw P(yes); ``verdict`` applies the current yes/no band. Callers may
    apply any band of their own to ``probability``. ``request_sha256`` identifies the masked request
    that answered it, also when the answer came from the store."""

    item: Mapping
    probability: float
    verdict: NoulVerdict
    from_store: bool
    request_sha256: str


@dataclass(frozen=True)
class PickResult:
    """The whole distribution is kept: ``probabilities`` per option and the API's ``confidence``.
    ``confident`` applies the current threshold; callers may apply any rule of their own instead."""

    choice: str
    confidence: float
    probabilities: Mapping[str, float]
    confident: bool
    request_sha256: str


@dataclass(frozen=True)
class ScoreResult:
    score: float
    probabilities: Mapping[str, float]
    confidence: float
    request_sha256: str


@dataclass(frozen=True)
class AllAnswers:
    """Typed results by question name, each with its raw probabilities."""

    checks: Mapping[str, CheckResult]
    picks: Mapping[str, PickResult]
    scores: Mapping[str, ScoreResult]
    request_sha256: str


@dataclass(frozen=True)
class CallOffer:
    """One operation Jev may call, with the closed set of inputs code found for it this step."""

    name: str
    description: str
    argument: Pick
    options: Mapping[str, str]


@dataclass(frozen=True)
class CallDecision:
    operation: str
    argument: str
    route: PickResult
    argument_pick: PickResult

    @property
    def confident(self) -> bool:
        return self.route.confident and self.argument_pick.confident

    @property
    def request_sha256(self) -> str:
        return self.route.request_sha256


class Judge:
    """``calls`` counts requests sent (store hits are free). ``scope()`` gives one caller, such as a
    single search, its own counter on the same client, store and journal; every scope adds its calls
    to its parent, and ``max_calls`` caps a judge together with all of its scopes."""

    def __init__(
        self,
        client: JevClient | AsyncJevClient,
        *,
        masker: Masker | None = _DEFAULT_MASKER,
        scanner: Scanner | None = _DEFAULT_SCANNER,
        store: AnswerStore | None = None,
        thresholds: Thresholds | None = None,
        served_model: str | None = None,
        journal: Journal | None = None,
        max_calls: int | None = None,
    ) -> None:
        self.client = client
        self.masker = masker
        self.scanner = scanner
        self.store = store
        self.thresholds = thresholds or Thresholds()
        self.served_model = served_model
        self.journal = journal
        self.max_calls = max_calls
        self.calls = 0
        self.input_total = TokenTotal()
        self._parent: Judge | None = None
        self._bookkeeping = threading.Lock()

    def scope(self) -> Judge:
        child = copy.copy(self)
        child.max_calls = None
        child.calls = 0
        child.input_total = TokenTotal()
        child._parent = self
        return child

    def calls_left(self) -> int | None:
        """The calls this judge may still send under its own and its parents' caps; None when uncapped."""
        caps = [judge.max_calls - judge.calls for judge in self._chain() if judge.max_calls is not None]
        return max(0, min(caps)) if caps else None

    def cancel(self) -> None:
        """Ask a client with an owned cancellation boundary to abort its active requests."""
        cancel = getattr(self.client, "cancel", None)
        if cancel is not None:
            cancel()

    def effective(self, *overrides: Mapping[str, float] | None) -> Thresholds:
        thresholds = self.thresholds
        for layer in overrides:
            thresholds = thresholds.updated(layer)
        return thresholds

    def check_each(
        self,
        check: Check,
        items: Sequence[Mapping],
        shared: Mapping | None = None,
        *,
        list_name: str = "items",
        thresholds: Thresholds | None = None,
    ) -> list[CheckResult]:
        """One yes/no answer per item, batched into as few requests as the state size allows.
        Items already judged by the same question and model come from the store."""
        return self.check_every([check], items, shared, list_name=list_name, thresholds=thresholds)[
            check.name
        ]

    def check_every(
        self,
        checks: Sequence[Check],
        items: Sequence[Mapping],
        shared: Mapping | None = None,
        *,
        list_name: str = "items",
        thresholds: Thresholds | None = None,
        batch_budget: int | None = None,
    ) -> Mapping[str, list[CheckResult]]:
        """Every check asked about every item, in as few requests as the size budget allows.

        Independent checks about the same items travel together: one request per batch that fits,
        each carrying every check for every item in it, instead of one round trip per check over the
        whole list. A batch the provider would refuse for its input size is split by item and the
        halves measured again, so every request sent fits the measured input budget. Items already
        judged by the same question and model come from the store.
        """
        plan = self._check_plan(checks, items, shared, list_name, thresholds, batch_budget)
        for batch in plan.batches:
            for sub_batch, response in self._send_positions(plan, sorted(set(batch.slots.values()))):
                plan.answer(sub_batch, response)
        return plan.answers()

    def iter_check_each(
        self,
        check: Check,
        items: Sequence[Mapping],
        shared: Mapping | None = None,
        *,
        list_name: str = "items",
        thresholds: Thresholds | None = None,
    ) -> Iterator[CheckResult]:
        """Yield cached answers, then completed batches, preserving work before a later stop.

        Uses the same packing and cache as ``check_each``. Results arrive in completion order,
        not input order. A call-cap or provider failure still raises after earlier results yield.
        """
        for _name, result in self.iter_check_every(
            [check], items, shared, list_name=list_name, thresholds=thresholds
        ):
            yield result

    def iter_check_every(
        self,
        checks: Sequence[Check],
        items: Sequence[Mapping],
        shared: Mapping | None = None,
        *,
        list_name: str = "items",
        thresholds: Thresholds | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[str, CheckResult]]:
        """Every check asked about every item, yielded per answered question as batches complete.

        The streaming form of ``check_every``: it uses the same packing, splitting and cache, yields
        each result under the name of the check that asked for it, and yields cached answers before the
        first batch so store hits consume no live call. A call-cap failure, or a provider input-budget
        refusal that no split can answer, still raises after earlier results yield, so a caller keeps
        every answered batch.
        Cancellation is checked before each live request, including between split halves; cached and
        already answered results still
        yield in full. It does not cancel a request already in flight.
        """
        plan = self._check_plan(checks, items, shared, list_name, thresholds)
        names = {check.question_id: check.name for check in checks}
        for (position, question_id), answer in sorted(plan.answered.items()):
            yield names[question_id], plan.result(position, answer)
        for batch in plan.batches:
            sent = self._send_positions(plan, sorted(set(batch.slots.values())), cancelled=cancelled)
            for sub_batch, response in sent:
                plan.answer(sub_batch, response)
                for question_id, position in sorted(sub_batch.slots.items(), key=lambda slot: slot[1]):
                    yield (
                        names[question_id.split("#", 1)[0]],
                        plan.result(position, plan.answered[(position, question_id)]),
                    )

    async def check_each_async(
        self,
        check: Check,
        items: Sequence[Mapping],
        shared: Mapping | None = None,
        *,
        list_name: str = "items",
        thresholds: Thresholds | None = None,
    ) -> list[CheckResult]:
        """``check_each`` with its batches sent concurrently."""
        answers = await self.check_every_async(
            [check], items, shared, list_name=list_name, thresholds=thresholds
        )
        return answers[check.name]

    async def check_every_async(
        self,
        checks: Sequence[Check],
        items: Sequence[Mapping],
        shared: Mapping | None = None,
        *,
        list_name: str = "items",
        thresholds: Thresholds | None = None,
        batch_budget: int | None = None,
    ) -> Mapping[str, list[CheckResult]]:
        """``check_every`` with its batches sent concurrently; when the served model is still
        unknown and an answer store is present, the first batch pins the model before the remaining
        batches look in the store, exactly as the sequential path does."""
        plan = self._check_plan(checks, items, shared, list_name, thresholds, batch_budget)
        batches = plan.batches
        if batches and self.store is not None and not self._knows_model():
            # Without a served model the store cannot prove model-version identity, so every lookup
            # misses; the first live response pins ``served_model`` and the rest may then replay.
            # ``slots`` maps one entry per open question to its item position, so the positions are
            # deduplicated exactly like the sibling batch senders: two checks over two items are two
            # item slots and four questions, never four copies of two items and eight questions.
            for sub_batch, response in await self._send_positions_async(
                plan, sorted(set(batches[0].slots.values()))
            ):
                plan.answer(sub_batch, response)
            batches = batches[1:]
        await asyncio.gather(*(self._answer_batch_async(plan, batch) for batch in batches))
        return plan.answers()

    async def _answer_batch_async(self, plan: _CheckPlan, batch: _Batch) -> None:
        """Send one packed batch concurrently, splitting it when the provider refuses its size."""
        for sub_batch, response in await self._send_positions_async(plan, sorted(set(batch.slots.values()))):
            plan.answer(sub_batch, response)

    def ask_all(
        self,
        state: Mapping,
        *,
        checks: Sequence[Check] = (),
        picks: Sequence[tuple[Pick, Mapping[str, str]]] = (),
        scores: Sequence[Rate] = (),
        thresholds: Thresholds | None = None,
    ) -> AllAnswers:
        """Many independent questions over one state, in one request."""
        request = self._all_request(checks, picks, scores, thresholds)
        return request.answers(state, self.ask(state, request.questions, thresholds=request.thresholds))

    async def ask_all_async(
        self,
        state: Mapping,
        *,
        checks: Sequence[Check] = (),
        picks: Sequence[tuple[Pick, Mapping[str, str]]] = (),
        scores: Sequence[Rate] = (),
        thresholds: Thresholds | None = None,
    ) -> AllAnswers:
        request = self._all_request(checks, picks, scores, thresholds)
        response = await self.ask_async(state, request.questions, thresholds=request.thresholds)
        return request.answers(state, response)

    def pick(
        self, pick: Pick, options: Mapping[str, str], state: Mapping, *, thresholds: Thresholds | None = None
    ) -> PickResult | None:
        """The option Jev picks, or None when masking left no option to offer."""
        questions = self._pick_questions(pick, options)
        if questions is None:
            return None
        thresholds = thresholds or self.thresholds
        return _pick_result(self.ask(state, questions, thresholds=thresholds), pick.question_id, thresholds)

    async def pick_async(
        self, pick: Pick, options: Mapping[str, str], state: Mapping, *, thresholds: Thresholds | None = None
    ) -> PickResult | None:
        questions = self._pick_questions(pick, options)
        if questions is None:
            return None
        thresholds = thresholds or self.thresholds
        response = await self.ask_async(state, questions, thresholds=thresholds)
        return _pick_result(response, pick.question_id, thresholds)

    def choose_call(
        self,
        route: Pick,
        offers: Sequence[CallOffer],
        state: Mapping,
        *,
        thresholds: Thresholds | None = None,
    ) -> CallDecision | None:
        """Function calling in one request: which operation to run, and each operation's input,
        asked for every operation at once; only the chosen operation's answer is read."""
        request = self._call_request(route, offers, thresholds)
        if request is None:
            return None
        return request.decision(self.ask(state, request.questions, thresholds=request.thresholds))

    async def choose_call_async(
        self,
        route: Pick,
        offers: Sequence[CallOffer],
        state: Mapping,
        *,
        thresholds: Thresholds | None = None,
    ) -> CallDecision | None:
        request = self._call_request(route, offers, thresholds)
        if request is None:
            return None
        return request.decision(await self.ask_async(state, request.questions, thresholds=request.thresholds))

    def ask(
        self,
        state: Mapping,
        questions: Mapping,
        *,
        thresholds: Thresholds,
        item_keys: Mapping[str, str] | None = None,
        sources: Mapping[str, Mapping] | None = None,
        skeleton: Mapping | None = None,
    ) -> JevResponse:
        """Masks, scans, hashes and looks up the store; only a miss sends, and every fresh answer is
        recorded. The async variant shares every step except the send."""
        prepared = self._prepare(state, questions)
        if prepared.stored is not None:
            return prepared.stored
        self._reserve_call()
        dispatched = self._dispatch(prepared)
        return self._finish(prepared, dispatched, thresholds, item_keys, sources, skeleton)

    async def ask_async(
        self,
        state: Mapping,
        questions: Mapping,
        *,
        thresholds: Thresholds,
        item_keys: Mapping[str, str] | None = None,
        sources: Mapping[str, Mapping] | None = None,
        skeleton: Mapping | None = None,
    ) -> JevResponse:
        prepared = self._prepare(state, questions)
        if prepared.stored is not None:
            return prepared.stored
        self._reserve_call()
        dispatched = await self._dispatch_async(prepared)
        return self._finish(prepared, dispatched, thresholds, item_keys, sources, skeleton)

    def _prepare(self, state: Mapping, questions: Mapping) -> _Prepared:
        hidden: frozenset[str] = frozenset()
        if self.masker:
            state, questions, hidden = mask_request(state, questions, self.masker)
        refuse_if_secret(state, questions, self.scanner, hidden)
        request_hash = request_sha256(state, questions)
        stored = self.store.by_request(request_hash) if self.store else None
        accepted = stored.response() if stored is not None and self._accepts(stored.model) else None
        return _Prepared(state, questions, request_hash, request_body(state, questions), accepted)

    def _finish(
        self,
        prepared: _Prepared,
        dispatched: _Dispatched,
        thresholds: Thresholds,
        item_keys: Mapping[str, str] | None,
        sources: Mapping[str, Mapping] | None,
        skeleton: Mapping | None,
    ) -> JevResponse:
        response = dispatched.response
        with self._bookkeeping:
            for judge in self._chain():
                judge.served_model = response.model
                judge.input_total.add(response.input_tokens)
            self._record(prepared, dispatched, thresholds, item_keys or {}, sources or {}, skeleton or {})
        return JevResponse(response.answers, response.model, response.input_tokens, prepared.request_hash)

    def _reserve_call(self) -> None:
        with self._bookkeeping:
            if self.calls_left() == 0:
                raise CallCapReachedError("the call cap of this judge is used up")
            for judge in self._chain():
                judge.calls += 1

    def _chain(self) -> Iterator[Judge]:
        judge: Judge | None = self
        while judge is not None:
            yield judge
            judge = judge._parent

    def _dispatch(self, prepared: _Prepared) -> _Dispatched:
        if _is_async(self.client):
            raise TypeError("this Jev client is async; use the judge's *_async methods")
        request_id = self._journal_request(prepared)
        raw: RawResponse | None = None
        try:
            if hasattr(self.client, "send"):
                raw = self._send_with_attempt_callback(prepared, request_id)
                self._journal_response(request_id, raw)
                return _Dispatched.from_raw(self.client.parse(raw), raw, prepared)
            response = self.client.ask(prepared.state, prepared.questions)
            self._journal_response(request_id, RawResponse.from_decoded(response_to_raw(response)))
            return _Dispatched(response, prepared.body, sent_exact=False)
        except Exception as error:
            if isinstance(error, AttemptJournalCallbackError):
                self._propagate_attempt_journal_error(request_id, error, raw)
            self._journal_failure(request_id, error, raw)
            raise

    async def _dispatch_async(self, prepared: _Prepared) -> _Dispatched:
        """Awaits an async client; a sync client runs in a worker thread."""
        request_id = self._journal_request(prepared)
        raw: RawResponse | None = None
        try:
            if hasattr(self.client, "send"):
                sender = getattr(self.client, "send_with_attempts", None)
                callback = self._attempt_callback(request_id)
                if callable(sender) and callback is not None:
                    raw = await _awaited(sender, prepared.state, prepared.questions, on_attempt=callback)
                else:
                    raw = await _awaited(self.client.send, prepared.state, prepared.questions)
                self._journal_response(request_id, raw)
                return _Dispatched.from_raw(self.client.parse(raw), raw, prepared)
            response = await _awaited(self.client.ask, prepared.state, prepared.questions)
            self._journal_response(request_id, RawResponse.from_decoded(response_to_raw(response)))
            return _Dispatched(response, prepared.body, sent_exact=False)
        except Exception as error:
            if isinstance(error, AttemptJournalCallbackError):
                self._propagate_attempt_journal_error(request_id, error, raw)
            self._journal_failure(request_id, error, raw)
            raise

    def _send_positions(
        self,
        plan: _CheckPlan,
        positions: Sequence[int],
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[_Batch, JevResponse]]:
        """Send one packed batch of item positions, splitting it when its request cannot fit the
        provider's measured input budget.

        The batch is measured before it is sent: a request outside ``request_exceeds_input_budget``
        (state plus the longest question over ``JEV_INPUT_BOX_CHARS``, or the body over
        ``MAX_REQUEST_CHARS``) is split by item and each half is measured again, so no request the
        measurement already rejects is ever paid for. A provider refusal that still names an
        exceeded input budget (``max_tokens_exceeded``) splits the same way. One position whose own
        state cannot fit has no smaller honest request - its questions name an item path that a
        partial state would change - so its error propagates and the journal keeps the provider's
        report. Every sub-batch keeps each item's store key, so replay and resume accounting stay
        exact.
        """
        if not positions:
            return
        batch = self._batch(plan, sorted(positions))
        if batch is None:
            return
        splittable = len(positions) > 1
        if splittable and request_exceeds_input_budget(batch.state, batch.questions):
            yield from self._split_positions(plan, positions, cancelled)
            return
        try:
            if cancelled is not None and cancelled():
                return
            yield batch, self.ask(batch.state, batch.questions, thresholds=plan.thresholds, **batch.extras)
        except InputBudgetExceededError:
            if not splittable:
                raise
            yield from self._split_positions(plan, positions, cancelled)

    def _split_positions(
        self,
        plan: _CheckPlan,
        positions: Sequence[int],
        cancelled: Callable[[], bool] | None = None,
    ) -> Iterator[tuple[_Batch, JevResponse]]:
        middle = len(positions) // 2
        yield from self._send_positions(plan, positions[:middle], cancelled)
        yield from self._send_positions(plan, positions[middle:], cancelled)

    async def _send_positions_async(
        self, plan: _CheckPlan, positions: Sequence[int]
    ) -> list[tuple[_Batch, JevResponse]]:
        """The async form of ``_send_positions``; halves recurse sequentially so a refusal cannot
        leave an orphaned half running inside a cancelled gather."""
        if not positions:
            return []
        batch = self._batch(plan, sorted(positions))
        if batch is None:
            return []
        splittable = len(positions) > 1
        if splittable and request_exceeds_input_budget(batch.state, batch.questions):
            return await self._split_positions_async(plan, positions)
        try:
            response = await self.ask_async(
                batch.state, batch.questions, thresholds=plan.thresholds, **batch.extras
            )
        except InputBudgetExceededError:
            if not splittable:
                raise
            return await self._split_positions_async(plan, positions)
        return [(batch, response)]

    async def _split_positions_async(
        self, plan: _CheckPlan, positions: Sequence[int]
    ) -> list[tuple[_Batch, JevResponse]]:
        middle = len(positions) // 2
        left = await self._send_positions_async(plan, positions[:middle])
        right = await self._send_positions_async(plan, positions[middle:])
        return [*left, *right]

    def _journal_request(self, prepared: _Prepared) -> str | None:
        if self.journal is None:
            return None
        request = JournalRequest(
            prepared.request_hash, self.client.model, prepared.state, prepared.questions, prepared.body
        )
        return self.journal.record_request(request)

    def _journal_response(self, request_id: str | None, raw: RawResponse) -> None:
        if self.journal is not None and request_id is not None:
            self.journal.record_response(request_id, raw)

    def _journal_failure(self, request_id: str | None, error: Exception, raw: RawResponse | None) -> None:
        if self.journal is not None and request_id is not None:
            self.journal.record_failure(request_id, f"{type(error).__name__}: {error}", raw)

    def _propagate_attempt_journal_error(
        self, request_id: str | None, error: AttemptJournalCallbackError, raw: RawResponse | None
    ) -> None:
        original = error.original_error
        try:
            self._journal_failure(request_id, original, raw)
        except Exception as failure_error:
            original.add_note(
                "The logical request failure could not be recorded either: "
                f"{type(failure_error).__name__}: {failure_error}"
            )
        raise original from error

    def _send_with_attempt_callback(self, prepared: _Prepared, request_id: str | None) -> RawResponse:
        sender = getattr(self.client, "send_with_attempts", None)
        callback = self._attempt_callback(request_id)
        if callable(sender) and callback is not None:
            return sender(prepared.state, prepared.questions, on_attempt=callback)
        return self.client.send(prepared.state, prepared.questions)

    def _attempt_callback(self, request_id: str | None):
        if self.journal is None or request_id is None:
            return None
        record_attempt = getattr(self.journal, "record_attempt", None)
        if not callable(record_attempt):
            return None
        return lambda attempt: record_attempt(request_id, attempt)

    def _check_plan(
        self,
        checks: Sequence[Check],
        items: Sequence[Mapping],
        shared: Mapping | None,
        list_name: str,
        thresholds: Thresholds | None,
        batch_budget: int | None = None,
    ) -> _CheckPlan:
        """Mask the whole candidate set before packing so copied secret values stay hidden across
        batches. Each per-item store key includes its masked item and shared state; a batch also
        masks its question wording with those values before sending.
        """
        if len({check.name for check in checks}) != len(checks):
            raise ValueError("independent checks require unique names for their result lists")
        *items, shared = self._masked_together([*items, shared or {}])
        budget = MAX_STATE_CHARS if batch_budget is None else batch_budget
        plan = _CheckPlan(list_name, checks, items, shared, thresholds or self.thresholds, budget)
        for position, item in enumerate(items):
            for check in checks:
                stored = self._stored_item(check, item, shared)
                if stored is None:
                    plan.open.setdefault(position, {})[check.question_id] = check
                else:
                    plan.answered[(position, check.question_id)] = stored
        plan.batches = [
            batch
            for batch in (self._batch(plan, positions) for positions in _batches(plan))
            if batch is not None
        ]
        return plan

    def _masked_together(self, values: list[Mapping]) -> list[Mapping]:
        """Masks the values as one request: a value hidden in one of them is hidden in all."""
        if self.masker is None:
            return values
        hidden = masked_values(values, self.masker)
        return [mask_everywhere(value, self.masker, hidden) for value in values]

    def _batch(self, plan: _CheckPlan, positions: list[int]) -> _Batch | None:
        """The request that carries one batch: every still open check of every item in it, asked at
        that item's place in the list. An item whose questions were all answered before the call is
        in no batch, and an item that still needs one answer keeps its own key and its own source.
        """
        *batch_items, batch_shared = self._masked_together(
            [*(plan.items[position] for position in positions), plan.shared]
        )
        by_position = dict(zip(positions, batch_items, strict=True))
        questions: dict[str, dict] = {}
        slots: dict[str, int] = {}
        item_keys: dict[str, str] = {}
        sources: dict[str, dict] = {}
        for slot, position in enumerate(positions):
            item = by_position[position]
            for check in plan.open_at(position).values():
                asked = f"{check.question_id}#{slot}"
                questions[asked] = check.to_question(item_path(plan.list_name, slot))
                slots[asked] = position
                item_keys[self._item_key(check, item, batch_shared)] = asked
                if _source_of(item):
                    sources[asked] = _source_of(item)
        if not questions:
            return None
        extras = {
            "item_keys": item_keys,
            "sources": sources,
            "skeleton": _skeleton(plan.list_name, questions, batch_items, batch_shared),
        }
        return _Batch({**batch_shared, plan.list_name: batch_items}, questions, slots, extras)

    def _all_request(
        self,
        checks: Sequence[Check],
        picks: Sequence[tuple[Pick, Mapping[str, str]]],
        scores: Sequence[Rate],
        thresholds: Thresholds | None,
    ) -> _AllRequest:
        offered = {pick.name: safe_options(options, self.masker) for pick, options in picks}
        questions = {check.question_id: check.to_question() for check in checks}
        questions |= {pick.question_id: pick.to_question(offered[pick.name]) for pick, _ in picks}
        questions |= {rate.question_id: rate.to_question() for rate in scores}
        pick_questions = tuple(pick for pick, _ in picks)
        return _AllRequest(
            tuple(checks), pick_questions, tuple(scores), questions, thresholds or self.thresholds
        )

    def _pick_questions(self, pick: Pick, options: Mapping[str, str]) -> dict | None:
        offered = safe_options(options, self.masker)
        return {pick.question_id: pick.to_question(offered)} if offered else None

    def _call_request(
        self, route: Pick, offers: Sequence[CallOffer], thresholds: Thresholds | None
    ) -> _CallRequest | None:
        usable = {offer.name: offer for offer in offers if safe_options(offer.options, self.masker)}
        if not usable:
            return None
        questions = {
            ROUTE_QUESTION: route.to_question({name: offer.description for name, offer in usable.items()})
        }
        for name, offer in usable.items():
            questions[_argument_id(name, offer)] = offer.argument.to_question(
                safe_options(offer.options, self.masker)
            )
        return _CallRequest(usable, questions, thresholds or self.thresholds)

    def _stored_item(self, check: Check, item: Mapping, shared: Mapping) -> _ItemAnswer | None:
        if self.store is None or not self._knows_model():
            return None
        stored = self.store.by_item(self._item_key(check, item, shared), self._model_filter())
        if stored is None or not isinstance(stored.answer, NoulAnswer):
            return None
        return _ItemAnswer(stored.answer.probability, True, stored.request_sha256)

    def _item_key(self, check: Check, item: Mapping, shared: Mapping) -> str:
        """Item content, the shared state the question refers to, and the question with its wording."""
        return f"{content_hash(item)}|{content_hash(shared)}|{check.question_id}"

    def _accepts(self, stored_model: str) -> bool:
        return self._knows_model() and self._model_filter() in (None, stored_model)

    def _knows_model(self) -> bool:
        return self._replays_any_model() or self.served_model is not None

    def _model_filter(self) -> str | None:
        return None if self._replays_any_model() else self.served_model

    def _replays_any_model(self) -> bool:
        return getattr(self.client, "replays_any_model", False)

    def _record(
        self,
        prepared: _Prepared,
        dispatched: _Dispatched,
        thresholds: Thresholds,
        item_keys: Mapping[str, str],
        sources: Mapping[str, Mapping],
        skeleton: Mapping,
    ) -> None:
        if self.store is None:
            return
        response = dispatched.response
        self.store.put(
            AnswerRecord(
                request_sha256=prepared.request_hash,
                question_ids=tuple(prepared.questions),
                answers={question_id: answer.to_json() for question_id, answer in response.answers.items()},
                model=response.model,
                input_tokens=response.input_tokens,
                thresholds=thresholds.as_dict(),
                item_keys=dict(item_keys),
                sources=dict(sources),
                skeleton=dict(skeleton),
                request={"state": prepared.state, "questions": prepared.questions},
                sent_body_base64=base64.b64encode(dispatched.sent_body).decode("ascii"),
                sent_exact=dispatched.sent_exact,
            )
        )


@dataclass(frozen=True)
class _ItemAnswer:
    probability: float
    from_store: bool
    request_sha256: str


@dataclass(frozen=True)
class _Prepared:
    """A masked, scanned and hashed request, with the stored answer when the store has one."""

    state: Mapping
    questions: Mapping
    request_hash: str
    body: bytes
    stored: JevResponse | None


@dataclass(frozen=True)
class _Dispatched:
    """The parsed answer and the request body that was sent: the wire bytes when the client's
    transport captured them (``sent_exact``), else the body as the library handed it over."""

    response: JevResponse
    sent_body: bytes
    sent_exact: bool

    @classmethod
    def from_raw(cls, response: JevResponse, raw: RawResponse, prepared: _Prepared) -> _Dispatched:
        if raw.sent_body is None:
            return cls(response, prepared.body, sent_exact=False)
        return cls(response, raw.sent_body, sent_exact=True)


@dataclass(frozen=True)
class _Batch:
    """One request: the code it carries, the questions asked about that code, and, for each of those
    questions, the item in the batch whose code the answer is about."""

    state: Mapping
    questions: Mapping
    slots: Mapping[str, int]
    extras: Mapping[str, Mapping]


@dataclass
class _CheckPlan:
    """The requests one judging call sends: each carries a batch of items and the questions still
    open for them, so independent questions about the same code share a request.

    ``answered`` and ``open`` are keyed by item position and question id, so a question answered
    before the call, or answered by an earlier batch, is never asked for again.
    """

    list_name: str
    checks: Sequence[Check]
    items: Sequence[Mapping]
    shared: Mapping
    thresholds: Thresholds
    budget: int
    answered: dict[tuple[int, str], _ItemAnswer] = field(default_factory=dict)
    open: dict[int, dict[str, Check]] = field(default_factory=dict)
    batches: list[_Batch] = field(default_factory=list)

    def open_at(self, position: int) -> Mapping[str, Check]:
        """The questions still open for one item, keyed by the question id without its slot."""
        return self.open.get(position, {})

    def pending(self) -> list[int]:
        """The items with a question still open for them, in the order they were given."""
        return sorted(self.open)

    def answer(self, batch: _Batch, response: JevResponse) -> None:
        for question_id, position in batch.slots.items():
            answer = response.noul(question_id)
            self.answered[(position, question_id)] = _ItemAnswer(
                answer.probability, response.from_store, response.request_sha256
            )

    def answers(self) -> dict[str, list[CheckResult]]:
        """The answers of this call, kept under the obligation that asked for them."""
        return {check.name: self.answers_for(check) for check in self.checks}

    def answers_for(self, check: Check) -> list[CheckResult]:
        asked = f"{check.name}@"
        return [
            self.result(position, answer)
            for (position, question_id), answer in sorted(self.answered.items())
            if question_id.startswith(asked)
        ]

    def result(self, position: int, answer: _ItemAnswer) -> CheckResult:
        return CheckResult(
            self.items[position],
            answer.probability,
            self.thresholds.noul_verdict(answer.probability),
            answer.from_store,
            answer.request_sha256,
        )


@dataclass(frozen=True)
class _AllRequest:
    checks: tuple[Check, ...]
    picks: tuple[Pick, ...]
    scores: tuple[Rate, ...]
    questions: Mapping
    thresholds: Thresholds

    def answers(self, state: Mapping, response: JevResponse) -> AllAnswers:
        return AllAnswers(
            {check.name: _check_result(response, check, state, self.thresholds) for check in self.checks},
            {pick.name: _pick_result(response, pick.question_id, self.thresholds) for pick in self.picks},
            {rate.name: _score_result(response, rate.question_id) for rate in self.scores},
            response.request_sha256,
        )


@dataclass(frozen=True)
class _CallRequest:
    usable: Mapping[str, CallOffer]
    questions: Mapping
    thresholds: Thresholds

    def decision(self, response: JevResponse) -> CallDecision:
        route_result = _pick_result(response, ROUTE_QUESTION, self.thresholds)
        chosen = self.usable[route_result.choice]
        argument_result = _pick_result(response, _argument_id(chosen.name, chosen), self.thresholds)
        return CallDecision(chosen.name, argument_result.choice, route_result, argument_result)


def _is_async(client: object) -> bool:
    method = getattr(client, "send", None) or client.ask
    return inspect.iscoroutinefunction(method)


async def _awaited(method, *arguments, **keywords):
    if inspect.iscoroutinefunction(method):
        return await method(*arguments, **keywords)
    return await asyncio.to_thread(method, *arguments, **keywords)


def _check_result(response: JevResponse, check: Check, state: Mapping, thresholds: Thresholds) -> CheckResult:
    probability = response.noul(check.question_id).probability
    return CheckResult(
        state, probability, thresholds.noul_verdict(probability), response.from_store, response.request_sha256
    )


def _score_result(response: JevResponse, question_id: str) -> ScoreResult:
    answer = response.score(question_id)
    return ScoreResult(answer.score, answer.probabilities, answer.confidence, response.request_sha256)


def _skeleton(list_name: str, questions: Mapping, items: list[Mapping], shared: Mapping) -> dict:
    """Everything needed to rebuild a batched request except the code itself and the shared state:
    the code is re-read from each item's file and lines, and only hashes of it are kept."""
    return {
        "list_name": list_name,
        "questions": dict(questions),
        "items": [{key: value for key, value in item.items() if key != CODE_FIELD} for item in items],
        "item_code_sha256": [content_hash(item.get(CODE_FIELD, "")) for item in items],
        "shared_sha256": content_hash(shared),
    }


def _source_of(item: Mapping) -> dict:
    """The location fields an item carries (``file``, ``lines``, ``commit``), if any."""
    return {key: item[key] for key in ("file", "lines", "commit") if key in item}


def _pick_result(response: JevResponse, question_id: str, thresholds: Thresholds) -> PickResult:
    answer = response.choice(question_id)
    return PickResult(
        answer.choice,
        answer.confidence,
        answer.probabilities,
        thresholds.choice_is_confident(answer.confidence),
        response.request_sha256,
    )


def _argument_id(operation: str, offer: CallOffer) -> str:
    return f"{operation}.{offer.argument.question_id}"


def _batches(plan: _CheckPlan) -> list[list[int]]:
    """Fills a batch until the request that would carry it would be larger than the budget allows.
    An item is measured together with the question wording asked about it, because that is what one
    request has to fit, and a single question is measured exactly as the batch around it is.
    """
    budget = plan.budget - len(json.dumps(plan.shared))
    batches: list[list[int]] = []
    current: list[int] = []
    used = 0
    for position in plan.pending():
        size = _open_size(plan, position)
        if current and used + size > budget:
            batches.append(current)
            current, used = [], 0
        current.append(position)
        used += size
    return [*batches, current] if current else batches


def _open_size(plan: _CheckPlan, position: int) -> int:
    """What one item and the wording of the questions still open about it would cost on their own."""
    wording = [check.to_question(item_path(plan.list_name, 0)) for check in plan.open_at(position).values()]
    return len(json.dumps(plan.items[position])) + sum(len(json.dumps(question)) for question in wording)
