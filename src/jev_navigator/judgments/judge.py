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
import threading
from collections.abc import AsyncIterator, Callable, Generator, Iterator, Mapping, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor, as_completed, wait
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from .answers import JevResponse, NoulAnswer, TokenTotal, response_to_raw
from .client import (
    JEV_INPUT_BOX_CHARS,
    MAX_REQUEST_CHARS,
    AsyncJevClient,
    InputBudgetExceededError,
    JevClient,
    UnansweredQuestionError,
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

DEFAULT_ITEMS_PER_REQUEST = 16
"""How many items one batched request carries at most (André, 03.10.2026: measured on the code-index
set, 16 per request kept accuracy and cost about half the tokens of one per request)."""
BATCHING_RULE = "unit-place-order-count-and-box-v1"
"""How batches form: units in file-and-lines order, closed at ``items_per_request`` items or at the
character box. Recorded on every stored answer; the batch membership hash in the item key already
tells two batches apart."""
DEFAULT_MAX_CONCURRENCY = 16
"""How many requests one judge, together with all of its scopes, has in flight at once, on the
sync and the async path."""
SEND_SLOT_POLL_SECONDS = 0.01
"""How often an async send waiting for a free slot looks again. The slots are a thread semaphore,
shared with sync sends; waiting on one from the event loop by polling never blocks the loop and
never takes an executor thread that a sync client's send needs to finish."""
CODE_FIELD = "code"
ROUTE_QUESTION = "route"
_DEFAULT_MASKER = SecretMasker()
_DEFAULT_SCANNER = SecretScanner()


ABORTED_SEND_ERRORS: tuple[type[BaseException], ...] = (CancelledError, KeyboardInterrupt)
"""What a send ends with when ``Judge.abort_sends`` stopped it: the client's or the pool's
``CancelledError``, or the caller's interrupt itself. Any other error is a real failure."""


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
    """``calls`` counts requests sent (store hits are free); ``replayed_answers`` counts the answers
    the store gave instead. ``scope()`` gives one caller, such as a
    single search, its own counter on the same client, store and journal; every scope adds its calls
    to its parent, and ``max_calls`` caps a judge together with all of its scopes.
    ``items_per_request`` caps the items of one batched request, and ``max_concurrency`` bounds how
    many requests this judge and all of its scopes have in flight, sync or async, whichever callers
    send them: a search's beam, its nested batches and history checks draw on the same slots. A slot
    is held only around the client's send, never around an opening or a whole batched call, so a
    caller holding none can always wait on its own nested requests."""

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
        items_per_request: int = DEFAULT_ITEMS_PER_REQUEST,
        max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    ) -> None:
        self.client = client
        self.masker = masker
        self.scanner = scanner
        self.store = store
        self.thresholds = thresholds or Thresholds()
        self.served_model = served_model
        self.journal = journal
        self.max_calls = max_calls
        self.items_per_request = _at_least_one("items_per_request", items_per_request)
        self.max_concurrency = _at_least_one("max_concurrency", max_concurrency)
        self.calls = 0
        self.replayed_answers = 0
        self.input_total = TokenTotal()
        self._parent: Judge | None = None
        self._bookkeeping = threading.Lock()
        self._send_slots = threading.BoundedSemaphore(self.max_concurrency)

    def scope(self) -> Judge:
        child = copy.copy(self)
        child.max_calls = None
        child.calls = 0
        child.replayed_answers = 0
        child.input_total = TokenTotal()
        child._parent = self
        child._send_slots = self._send_slots
        return child

    @property
    def unanswered_requests(self) -> int:
        """The requests sent whose response never arrived, so whose token usage is unknown."""
        return self.calls - self.input_total.responses

    def calls_left(self) -> int | None:
        """The calls this judge may still send under its own and its parents' caps; None when uncapped."""
        caps = [judge.max_calls - judge.calls for judge in self._chain() if judge.max_calls is not None]
        return max(0, min(caps)) if caps else None

    def cancel(self) -> None:
        """Ask a client with an owned cancellation boundary to abort its active requests."""
        cancel = getattr(self.client, "cancel", None)
        if cancel is not None:
            cancel()

    def abort_sends(self, futures: Sequence[Future]) -> None:
        """What a caller interrupt does to one step's concurrent requests: the client aborts those in
        flight, those not started never start, and this returns once every one has settled. Only an
        interrupt calls it. A client's cancel is permanent (``TypeSafeJevClient`` refuses every later
        request once cancelled), so cancelling on an ordinary failure, such as a reached call cap,
        would fail the rest of the run and discard answers already paid for; an ordinary failure
        instead lets the requests in flight settle and keeps their answers. Known and accepted: a
        Ctrl-C that lands while a caller handles a yielded answer closes the batch generator, which
        cannot tell it from an ordinary early stop, so the requests in flight finish first and only
        the exit is delayed."""
        self.cancel()
        for future in futures:
            future.cancel()
        wait(futures)

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
    ) -> Mapping[str, list[CheckResult]]:
        """Every check asked about every item, in as few requests as the size budget allows.

        Independent checks about the same items travel together: one request per batch that fits,
        each carrying every check for every item in it, instead of one round trip per check over the
        whole list. Batches hold at most ``items_per_request`` items and form in a stable item order,
        so the same items always form the same batches; an item's stored answer is reused only
        with the same batch mates. Batches are sent concurrently, within the judge-wide
        ``max_concurrency``.
        A batch the provider would refuse for its input size is split by item and the halves
        measured again, so every request sent fits the measured input budget. Items already judged
        by the same question and model, in the same batch, come from the store.
        """
        plan = self._check_plan(checks, items, shared, list_name, thresholds)
        for sub_batch, response in self._answered_batches(plan):
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

        The streaming form of ``check_every``: it uses the same packing, splitting, concurrency and
        cache, yields each result under the name of the check that asked for it as its batch
        completes, and yields cached answers before the first batch so store hits consume no live
        call. A call-cap failure, or a provider input-budget refusal that no split can answer, still
        raises after every answered batch has yielded, so a caller keeps them all.
        Cancellation is checked before each live request, including between split halves; cached and
        already answered results still yield in full. It does not cancel a request already in flight.
        """
        plan = self._check_plan(checks, items, shared, list_name, thresholds)
        names = {check.question_id: check.name for check in checks}
        for (position, question_id), answer in sorted(plan.answered.items()):
            yield names[question_id], plan.result(position, answer)
        for sub_batch, response in self._answered_batches(plan, cancelled):
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
    ) -> Mapping[str, list[CheckResult]]:
        """``check_every`` with its batches sent concurrently, at most ``max_concurrency`` at once (the
        judge-wide send slots bound them together with every other caller), and,
        under a call cap, in waves no larger than the calls left, as the sync path does. When the
        served model is still unknown and an answer store is present, the first batch pins the model
        before the remaining batches look in the store. A batch refused for its size comes back as
        its halves, which are sent one per wave before any other batch, as on the sync path. A failed
        batch stops the batches still waiting for a slot of this call; one already waiting for a
        judge-wide send slot still sends. The wave settles whole before its failure is raised, so no
        request of the call is still running when the error comes out."""
        plan = self._check_plan(checks, items, shared, list_name, thresholds)
        queue = _WaveQueue(list(plan.batches))
        slots = asyncio.Semaphore(self.max_concurrency)
        halted = asyncio.Event()
        while queue:
            wave = queue.next_wave(self._next_wave_size)
            settled = await asyncio.gather(
                *(self._answer_batch_async(plan, batch, slots, halted) for batch in wave),
                return_exceptions=True,
            )
            failure = _wave_failure([outcome for outcome in settled if isinstance(outcome, BaseException)])
            if failure is not None:
                raise failure
            queue.put_halves([half for halves in settled for half in halves])
        return plan.answers()

    async def _answer_batch_async(
        self, plan: _CheckPlan, batch: _Batch, slots: asyncio.Semaphore, halted: asyncio.Event
    ) -> list[_Batch]:
        """Send one packed batch once a slot is free and apply its answers; a size refusal returns
        its halves instead. After a sibling failed, a batch still waiting for a slot is not sent."""
        async with slots:
            if halted.is_set():
                return []
            try:
                response = await self.ask_async(
                    batch.state,
                    batch.questions,
                    thresholds=plan.thresholds,
                    masked=plan.hidden,
                    **batch.extras,
                )
            except InputBudgetExceededError:
                halves = self._halves(plan, batch)
                if halves is None:
                    halted.set()
                    raise
                return halves
            except Exception:
                halted.set()
                raise
        plan.answer(batch, response)
        return []

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
        masked: frozenset[str] | None = None,
        batch: Mapping | None = None,
    ) -> JevResponse:
        """Masks, scans, hashes and looks up the store; only a miss sends, and every fresh answer is
        recorded. The async variant shares every step except the send. ``masked`` marks a request
        the caller already masked as one, with the values it hid: masking is skipped, the final
        scan is not. Question wording must never quote customer text; a batch plan masks it anyway,
        once per plan."""
        prepared = self._sendable(state, questions, masked)
        if prepared.stored is not None:
            self._count_replayed(len(prepared.stored.answers))
            return prepared.stored
        self._reserve_call()
        dispatched = self._dispatch(prepared)
        return self._finish(prepared, dispatched, thresholds, item_keys, sources, skeleton, batch)

    async def ask_async(
        self,
        state: Mapping,
        questions: Mapping,
        *,
        thresholds: Thresholds,
        item_keys: Mapping[str, str] | None = None,
        sources: Mapping[str, Mapping] | None = None,
        skeleton: Mapping | None = None,
        masked: frozenset[str] | None = None,
        batch: Mapping | None = None,
    ) -> JevResponse:
        prepared = self._sendable(state, questions, masked)
        if prepared.stored is not None:
            self._count_replayed(len(prepared.stored.answers))
            return prepared.stored
        self._reserve_call()
        dispatched = await self._dispatch_async(prepared)
        return self._finish(prepared, dispatched, thresholds, item_keys, sources, skeleton, batch)

    def _sendable(self, state: Mapping, questions: Mapping, masked: frozenset[str] | None) -> _Prepared:
        """The prepared request, refused without a call when this route, under the same input box,
        already refused these exact bytes for their input size, so a replay splits it again for free.
        A changed box tries the request again."""
        prepared = self._prepare(state, questions, masked)
        if self._known_refusal(prepared):
            raise InputBudgetExceededError("this route refused this exact request for its input size before")
        return prepared

    def _known_refusal(self, prepared: _Prepared) -> bool:
        if prepared.stored is not None or self.store is None:
            return False
        return self.store.refused(prepared.request_hash, self.client.model, JEV_INPUT_BOX_CHARS)

    def _record_refusal(self, prepared: _Prepared, error: Exception) -> None:
        if isinstance(error, InputBudgetExceededError) and self.store is not None:
            with self._bookkeeping:
                self.store.put_refusal(prepared.request_hash, self.client.model, JEV_INPUT_BOX_CHARS)

    def _prepare(self, state: Mapping, questions: Mapping, masked: frozenset[str] | None) -> _Prepared:
        if masked is None:
            state, questions, masked = self._masked_request(state, questions)
        refuse_if_secret(state, questions, self.scanner, masked)
        request_hash = request_sha256(state, questions)
        stored = self._stored_request(request_hash)
        accepted = stored.response() if stored is not None else None
        return _Prepared(state, questions, request_hash, request_body(state, questions), accepted)

    def _masked_request(self, state: Mapping, questions: Mapping) -> tuple[Mapping, Mapping, frozenset[str]]:
        if not self.masker:
            return state, questions, frozenset()
        return mask_request(state, questions, self.masker)

    def _finish(
        self,
        prepared: _Prepared,
        dispatched: _Dispatched,
        thresholds: Thresholds,
        item_keys: Mapping[str, str] | None,
        sources: Mapping[str, Mapping] | None,
        skeleton: Mapping | None,
        batch: Mapping | None,
    ) -> JevResponse:
        response = dispatched.response
        with self._bookkeeping:
            self._record(
                prepared, dispatched, thresholds, item_keys or {}, sources or {}, skeleton or {}, batch or {}
            )
        return JevResponse(response.answers, response.model, response.input_tokens, prepared.request_hash)

    def _accepted(self, prepared: _Prepared, dispatched: _Dispatched) -> _Dispatched:
        """Counts the served model and the tokens a parsed response cost, then refuses one that left
        out an asked answer, so the send's failure handling journals it like any failed response."""
        response = dispatched.response
        with self._bookkeeping:
            for judge in self._chain():
                judge.served_model = response.model
                judge.input_total.add(response.input_tokens)
        _refuse_unanswered(prepared.questions, response)
        return dispatched

    def _count_replayed(self, answers: int) -> None:
        with self._bookkeeping:
            for judge in self._chain():
                judge.replayed_answers += answers

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
                with self._send_slots:
                    raw = self._send_with_attempt_callback(prepared, request_id)
                self._journal_response(request_id, raw)
                return self._accepted(prepared, _Dispatched.from_raw(self.client.parse(raw), raw, prepared))
            with self._send_slots:
                response = self.client.ask(prepared.state, prepared.questions)
            raw = RawResponse.from_decoded(response_to_raw(response))
            self._journal_response(request_id, raw)
            return self._accepted(prepared, _Dispatched(response, prepared.body, sent_exact=False))
        except Exception as error:
            if isinstance(error, AttemptJournalCallbackError):
                self._propagate_attempt_journal_error(request_id, error, raw)
            self._journal_failure(request_id, error, raw)
            self._record_refusal(prepared, error)
            raise

    async def _dispatch_async(self, prepared: _Prepared) -> _Dispatched:
        """Awaits an async client; a sync client runs in a worker thread."""
        request_id = self._journal_request(prepared)
        raw: RawResponse | None = None
        try:
            if hasattr(self.client, "send"):
                sender = getattr(self.client, "send_with_attempts", None)
                callback = self._attempt_callback(request_id)
                async with self._send_slot_async():
                    if callable(sender) and callback is not None:
                        raw = await _awaited(sender, prepared.state, prepared.questions, on_attempt=callback)
                    else:
                        raw = await _awaited(self.client.send, prepared.state, prepared.questions)
                self._journal_response(request_id, raw)
                return self._accepted(prepared, _Dispatched.from_raw(self.client.parse(raw), raw, prepared))
            async with self._send_slot_async():
                response = await _awaited(self.client.ask, prepared.state, prepared.questions)
            raw = RawResponse.from_decoded(response_to_raw(response))
            self._journal_response(request_id, raw)
            return self._accepted(prepared, _Dispatched(response, prepared.body, sent_exact=False))
        except Exception as error:
            if isinstance(error, AttemptJournalCallbackError):
                self._propagate_attempt_journal_error(request_id, error, raw)
            self._journal_failure(request_id, error, raw)
            self._record_refusal(prepared, error)
            raise

    @asynccontextmanager
    async def _send_slot_async(self) -> AsyncIterator[None]:
        """One of the judge-wide send slots, awaited without blocking the event loop."""
        while not self._send_slots.acquire(blocking=False):
            await asyncio.sleep(SEND_SLOT_POLL_SECONDS)
        try:
            yield
        finally:
            self._send_slots.release()

    def _must_learn_model_first(self) -> bool:
        """Without a served model the store cannot prove model-version identity, so every lookup
        misses; one batch sent alone pins ``served_model`` and the rest may then replay."""
        return self.store is not None and not self._knows_model()

    def _answered_batches(
        self, plan: _CheckPlan, cancelled: Callable[[], bool] | None = None
    ) -> Iterator[tuple[_Batch, JevResponse]]:
        """Every request of the plan's batches with its answer, a batch at a time as batches complete,
        with at most ``max_concurrency`` worker threads; their sends share the judge-wide send slots.
        Answers are applied by the caller, on its
        own thread. Under a call cap, batches go in waves no larger than the calls left, in their
        stable order, so a capped call always answers the same batches. A batch refused for its size
        comes back as its halves, which are sent one per wave before any other batch (see
        ``_WaveQueue``), so they never take a call from their own wave. The first failure stops
        every batch that has not started, and raises after every batch already started has yielded
        what it answered. A batch that started and is still waiting for a judge-wide send slot when
        the failure comes still sends."""
        queue = _WaveQueue(list(plan.batches))
        if not queue:
            return
        stop = _BatchStop(cancelled)
        pool = ThreadPoolExecutor(max_workers=min(self.max_concurrency, len(plan.batches)))
        futures: list[Future] = []
        try:
            while queue and not stop.requested():
                futures = []
                for batch in queue.next_wave(self._next_wave_size):
                    futures.append(pool.submit(self._send_batch, plan, batch, stop))
                queue.put_halves((yield from _completed_wave(futures)))
        except KeyboardInterrupt:
            stop.halted.set()
            self.abort_sends(futures)
            _raise_provider_failure(futures)
            raise
        finally:
            stop.halted.set()
            pool.shutdown(wait=True, cancel_futures=True)

    def _next_wave_size(self, waiting: int) -> int:
        """One batch alone while the served model is still to be learned; else every waiting batch
        when uncapped, else as many as calls are left, at least one, since a batch the store answers
        needs no call."""
        if self._must_learn_model_first():
            return 1
        left = self.calls_left()
        return waiting if left is None else max(1, min(left, waiting))

    def _send_batch(self, plan: _CheckPlan, batch: _Batch, stop: _BatchStop) -> _SentBatch:
        """One batch's request on a worker thread. A size refusal returns the batch's halves for a
        later wave; any other failure stops the other batches' unsent requests and is kept for the
        caller to raise."""
        if stop.requested():
            return _SentBatch([], None, [])
        try:
            response = self.ask(
                batch.state, batch.questions, thresholds=plan.thresholds, masked=plan.hidden, **batch.extras
            )
        except InputBudgetExceededError as error:
            halves = self._halves(plan, batch)
            if halves is None:
                stop.halted.set()
                return _SentBatch([], error, [])
            return _SentBatch([], None, halves)
        except Exception as error:
            stop.halted.set()
            return _SentBatch([], error, [])
        return _SentBatch([(batch, response)], None, [])

    def _halves(self, plan: _CheckPlan, batch: _Batch) -> list[_Batch] | None:
        """The two requests a refused batch splits into, by item, without the halves that have no
        open question; None for a one-item batch, whose request has no smaller honest form - its
        questions name an item path that a partial state would change."""
        members = list(batch.members)
        if len(members) < 2:
            return None
        middle = len(members) // 2
        built = (self._batch(plan, members[:middle]), self._batch(plan, members[middle:]))
        return [half for half in built if half is not None]

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
            self.journal.record_failure(request_id, _failure_text(error), raw)

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
    ) -> _CheckPlan:
        """Mask the whole candidate set once, before packing, so copied secret values stay hidden
        across batches. A value found in any check's wording is hidden in the items and shared state
        too, where it may stand without the context that marks it as secret. The final scan before
        each send still runs. Batches form over every item,
        answered or not, so they do not depend on the store. Each per-item store key includes its
        masked item, the shared state, the question and the batch it was asked in.
        """
        if len({check.name for check in checks}) != len(checks):
            raise ValueError("independent checks require unique names for their result lists")
        hidden = self._hidden_values(checks, items, shared or {})
        *items, shared = self._masked_together([*items, shared or {}], hidden)
        plan = _CheckPlan(
            list_name,
            checks,
            items,
            shared,
            hidden,
            thresholds or self.thresholds,
            self.items_per_request,
            masker=self.masker,
        )
        groups = _batches(plan)
        for members in groups:
            self._look_up_batch(plan, members)
        self._count_replayed(len(plan.answered))
        plan.batches = [
            batch for batch in (self._batch(plan, members) for members in groups) if batch is not None
        ]
        return plan

    def _hidden_values(
        self, checks: Sequence[Check], items: Sequence[Mapping], shared: Mapping
    ) -> frozenset[str]:
        if not self.masker:
            return frozenset()
        wordings = [check.to_question() for check in checks]
        return masked_values([*items, shared, *wordings], self.masker)

    def _look_up_batch(self, plan: _CheckPlan, members: list[int]) -> None:
        """Every question about every member is answered from the store or left open."""
        mates = plan.membership(members)
        for position in members:
            for check in plan.checks:
                stored = self._stored_item(check, plan.items[position], plan.shared, mates)
                if stored is None:
                    plan.open.setdefault(position, {})[check.question_id] = check
                else:
                    plan.answered[(position, check.question_id)] = stored

    def _masked_together(self, values: list[Mapping], hidden: frozenset[str]) -> list[Mapping]:
        """Masks the values as one request: each of ``hidden`` is hidden in all of them."""
        if self.masker is None:
            return values
        return [mask_everywhere(value, self.masker, hidden) for value in values]

    def _batch(self, plan: _CheckPlan, members: list[int]) -> _Batch | None:
        """The request that carries one batch: every member in the state, since an answer depends on
        its batch mates, and every still open check of every member, asked at that member's place
        in the list. A batch with no open question is not sent.
        """
        batch_items = [plan.items[position] for position in members]
        mates = plan.membership(members)
        questions: dict[str, dict] = {}
        slots: dict[str, int] = {}
        item_keys: dict[str, str] = {}
        sources: dict[str, dict] = {}
        for slot, (position, item) in enumerate(zip(members, batch_items, strict=True)):
            for check in plan.open_at(position).values():
                asked = f"{check.question_id}#{slot}"
                questions[asked] = plan.question(check, slot)
                slots[asked] = position
                item_keys[self._item_key(check, item, plan.shared, mates)] = asked
                if _source_of(item):
                    sources[asked] = _source_of(item)
        if not questions:
            return None
        extras = {
            "item_keys": item_keys,
            "sources": sources,
            "skeleton": _skeleton(plan.list_name, questions, batch_items, plan.shared),
            "batch": {
                "batching_rule": BATCHING_RULE,
                "items_per_request": plan.items_per_request,
                "members": [plan.item_ids[position] for position in members],
            },
        }
        return _Batch({**plan.shared, plan.list_name: batch_items}, questions, tuple(members), slots, extras)

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

    def _stored_item(self, check: Check, item: Mapping, shared: Mapping, mates: str) -> _ItemAnswer | None:
        if self.store is None or not self._knows_model():
            return None
        stored = self.store.by_item(self._item_key(check, item, shared, mates), self._model_filter())
        if stored is None or not isinstance(stored.answer, NoulAnswer):
            return None
        return _ItemAnswer(stored.answer.probability, True, stored.request_sha256)

    def _item_key(self, check: Check, item: Mapping, shared: Mapping, mates: str) -> str:
        """Item content, the shared state the question refers to, the question with its wording, and
        the batch the item was asked in."""
        return f"{content_hash(item)}|{content_hash(shared)}|{check.question_id}|{mates}"

    def _stored_request(self, request_hash: str) -> AnswerRecord | None:
        """The stored record this judge may replay for one request: none while its served model is
        unknown, else one from that model, or from any model for a replay-only client."""
        if self.store is None or not self._knows_model():
            return None
        return self.store.by_request(request_hash, self._model_filter())

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
        batch: Mapping,
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
                batch=dict(batch),
                request={"state": prepared.state, "questions": prepared.questions},
                sent_body_base64=base64.b64encode(dispatched.sent_body).decode("ascii"),
                sent_exact=dispatched.sent_exact,
            )
        )


def _refuse_unanswered(questions: Mapping, response: JevResponse) -> None:
    unanswered = [question_id for question_id in questions if question_id not in response.answers]
    if unanswered:
        raise UnansweredQuestionError(f"{response.model} returned no answer for {', '.join(unanswered)}")


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
class _BatchStop:
    """Why a concurrent batch must not send its next request: the caller cancelled, a sibling batch
    failed, or the caller stopped consuming results."""

    cancelled: Callable[[], bool] | None
    halted: threading.Event = field(default_factory=threading.Event)

    def requested(self) -> bool:
        return self.halted.is_set() or (self.cancelled is not None and self.cancelled())


class _WaveQueue:
    """The batches still to send, in stable order. The halves of a batch refused for its size are
    sent one per wave, before any other batch and depth first, as a sequential split would send
    them: so they never share a wave, and a call cap, with other batches, and a half that no split
    can answer stops its sibling from being sent."""

    def __init__(self, batches: list[_Batch]) -> None:
        self._batches = batches
        self._halves: list[_Batch] = []

    def __bool__(self) -> bool:
        return bool(self._halves or self._batches)

    def next_wave(self, size_for: Callable[[int], int]) -> list[_Batch]:
        if self._halves:
            return [self._halves.pop(0)]
        size = size_for(len(self._batches))
        wave, self._batches = self._batches[:size], self._batches[size:]
        return wave

    def put_halves(self, halves: list[_Batch]) -> None:
        self._halves = halves + self._halves


@dataclass(frozen=True)
class _SentBatch:
    """What one batch's worker brought back: its answer, the failure to raise, or the halves a size
    refusal split it into."""

    answered: list[tuple[_Batch, JevResponse]]
    error: Exception | None
    halves: list[_Batch]


@dataclass(frozen=True)
class _Batch:
    """One request: the code it carries, the questions asked about that code, and, for each of those
    questions, the item in the batch whose code the answer is about."""

    state: Mapping
    questions: Mapping
    members: tuple[int, ...]
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
    hidden: frozenset[str]
    thresholds: Thresholds
    items_per_request: int
    answered: dict[tuple[int, str], _ItemAnswer] = field(default_factory=dict)
    open: dict[int, dict[str, Check]] = field(default_factory=dict)
    batches: list[_Batch] = field(default_factory=list)
    item_ids: list[str] = field(init=False)
    masker: Masker | None = None
    _questions: dict[tuple[str, int], dict] = field(default_factory=dict, init=False)

    def open_at(self, position: int) -> Mapping[str, Check]:
        """The questions still open for one item, keyed by the question id without its slot."""
        return self.open.get(position, {})

    def __post_init__(self) -> None:
        self.item_ids = [content_hash(item) for item in self.items]

    def stable_order(self) -> list[int]:
        """Every item position ordered by the unit's place (file, then lines) where the item has one,
        else by its content, so the same units always form the same batches whatever order the
        caller gave them in, and a new commit does not reorder them."""
        return sorted(
            range(len(self.items)),
            key=lambda position: (*unit_place(self.items[position]), self.item_ids[position], position),
        )

    def question(self, check: Check, slot: int) -> dict:
        """The check's question about the item at ``slot``, masked with the plan's hidden values;
        built and masked once per plan, since packing measures many candidate requests."""
        key = (check.question_id, slot)
        if key not in self._questions:
            question = check.to_question(item_path(self.list_name, slot))
            self._questions[key] = (
                mask_everywhere(question, self.masker, self.hidden) if self.masker else question
            )
        return self._questions[key]

    def slot_questions(self, slot: int) -> list[tuple[str, dict]]:
        """Every check's question about the item at ``slot``, keyed as a request asks it."""
        return [(f"{check.question_id}#{slot}", self.question(check, slot)) for check in self.checks]

    def membership(self, members: Sequence[int]) -> str:
        """The identity of a batch: its members' content hashes in their order in the request."""
        return content_hash([self.item_ids[position] for position in members])

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


def _at_least_one(setting: str, value: int) -> int:
    if value < 1:
        raise ValueError(f"{setting} must be at least 1, got {value}")
    return value


def _completed_wave(futures: list[Future]) -> Generator[tuple[_Batch, JevResponse], None, list[_Batch]]:
    """Yield a wave's answers as its batches complete, then return the halves its refused batches
    split into, in the wave's stable order; after every batch settled, ``_wave_failure`` decides
    what is raised."""
    for future in as_completed(futures):
        yield from future.result().answered
    sent = [future.result() for future in futures]
    failure = _wave_failure([batch.error for batch in sent if batch.error is not None])
    if failure is not None:
        raise failure
    return [half for batch in sent for half in batch.halves]


def _wave_failure(failures: Sequence[BaseException]) -> BaseException | None:
    """The failure a settled wave raises: its first real failure, ahead of a call cap that only
    stopped a sibling, so a provider error is never reported as a spent budget. Every other failure
    of the wave is added to it as a note, so none is lost."""
    if not failures:
        return None
    primary = next((error for error in failures if not isinstance(error, CallCapReachedError)), failures[0])
    for other in failures:
        if other is not primary:
            primary.add_note(f"Also in this wave: {type(other).__name__}: {other}")
    return primary


def _raise_provider_failure(aborted: list[Future]) -> None:
    """The first real failure among settled batches an interrupt aborted, so it never lives only in
    the journal; the abort's own errors are not failures."""
    for future in aborted:
        if future.cancelled() or future.exception() is not None:
            continue
        error = future.result().error
        if error is not None and not isinstance(error, ABORTED_SEND_ERRORS):
            raise error


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


def unit_place(item: Mapping) -> tuple[str, int, int]:
    """The unit's file and line range, or empty when the item names none: the one order of judged
    units, which batches are packed in and results are listed in."""
    lines = item.get("lines")
    if not isinstance(lines, list | tuple) or len(lines) != 2:
        return str(item.get("file", "")), 0, 0
    return str(item.get("file", "")), int(lines[0]), int(lines[1])


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


def _failure_text(error: Exception) -> str:
    if isinstance(error, CancelledError) and not str(error):
        return f"{type(error).__name__}: the request was cancelled after it was sent"
    return f"{type(error).__name__}: {error}"


def _batches(plan: _CheckPlan) -> list[list[int]]:
    """Every item, in the stable order, in batches of at most ``items_per_request``; a batch closes
    early when the request carrying the next item would not fit the character boxes."""
    batches: list[list[int]] = []
    current: list[int] = []
    for position in plan.stable_order():
        if current and (
            len(current) == plan.items_per_request or not _fits_in_batch(plan, [*current, position])
        ):
            batches.append(current)
            current = []
        current.append(position)
    return [*batches, current] if current else batches


def _fits_in_batch(plan: _CheckPlan, members: list[int]) -> bool:
    """The one size rule of packing: the request that would carry these members, with every check
    asked of each, within the boxes ``request_exceeds_input_budget`` measures before a send."""
    state = {**plan.shared, plan.list_name: [plan.items[position] for position in members]}
    questions = {
        asked: question for slot in range(len(members)) for asked, question in plan.slot_questions(slot)
    }
    return not request_exceeds_input_budget(state, questions)
