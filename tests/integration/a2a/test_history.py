from __future__ import annotations

from uuid import uuid4

import pytest

from ant_ai.a2a.client import A2AClient
from ant_ai.a2a.config import A2AConfig
from ant_ai.core.events import Event
from tests.integration.a2a.conftest import (
    _drain_loop,
    _make_colony,
    _start_server,
    _stop_server,
    make_text_response,
)


async def _send_turn(
    client: A2AClient,
    message: str,
    *,
    context_id: str,
    reference_task_ids: list[str] | None = None,
) -> tuple[list[Event], str]:
    """Send one conversation turn; return ``(events, task_id)``.

    Asserts that at least one event carries a ``task_id`` — if this fails,
    the ``Event.task_id`` field was not added to the base class.
    """
    events: list[Event] = [
        ev
        async for ev in client.send_message(
            message,
            context_id=context_id,
            reference_task_ids=reference_task_ids,
        )
    ]
    task_id: str | None = next((e.task_id for e in events if e.task_id), None)
    assert task_id is not None, (
        "No task_id found in event stream. "
        "Make sure task_id is declared on the Event base class."
    )
    return events, task_id


def _client(port: int) -> A2AClient:
    return A2AClient(config=A2AConfig(endpoint=f"http://127.0.0.1:{port}/"))


def _all_text(messages: list) -> str:
    """Flatten all message content into a single string for assertions."""
    return " ".join(m.get("content", "") or "" for m in messages)


def _counting_dispatch(scripted_llm) -> list[int]:
    """Install a dispatch that records the number of messages per LLM call."""
    counts: list[int] = []

    async def dispatch(*, messages, **_):
        counts.append(len(messages))
        return make_text_response(f"Reply {len(counts)}.")

    scripted_llm.install(dispatch)
    return counts


def _capturing_dispatch(scripted_llm) -> dict[int, list]:
    """Install a dispatch that captures the full messages list per call number."""
    captured: dict[int, list] = {}

    async def dispatch(*, messages, **_):
        n = len(captured) + 1
        captured[n] = list(messages)
        return make_text_response(f"Reply {n}.")

    scripted_llm.install(dispatch)
    return captured


async def _assert_resume_continues_history(port: int, scripted_llm) -> None:
    """Turn 2 with reference_task_ids=[task_1_id] sees Q1+A1 in LLM messages."""
    captured: dict[int, list] = _capturing_dispatch(scripted_llm)
    ctx = str(uuid4())
    client: A2AClient = _client(port)

    _, task_1_id = await _send_turn(client, "What is your name?", context_id=ctx)
    await _send_turn(
        client, "What did I just ask?", context_id=ctx, reference_task_ids=[task_1_id]
    )
    await client.aclose()

    assert len(captured) == 2, f"Expected 2 LLM calls, got {len(captured)}"
    assert "What is your name?" in _all_text(captured[2]), (
        f"Turn-1 question not found in turn-2 LLM messages.\nMessages: {captured[2]}"
    )


async def _assert_conversations_are_isolated(
    port: int, scripted_llm, *, sentinel: str
) -> None:
    """Two separate context_ids do not bleed history into each other."""
    ctx_a, ctx_b = str(uuid4()), str(uuid4())

    async def dispatch_ok(*, messages, **_):
        return make_text_response("ok")

    scripted_llm.install(dispatch_ok)

    client: A2AClient = _client(port)
    _, _ = await _send_turn(client, f"{sentinel} is a secret.", context_id=ctx_a)
    _, task_b = await _send_turn(client, "Hello from B.", context_id=ctx_b)

    captured_b: list[list] = []

    async def dispatch_capture(*, messages, **_):
        captured_b.append(list(messages))
        return make_text_response("ok b")

    scripted_llm.install(dispatch_capture)

    await _send_turn(
        client, "What did I say?", context_id=ctx_b, reference_task_ids=[task_b]
    )
    await client.aclose()

    assert captured_b, "Follow-up dispatch was never called"
    assert sentinel not in _all_text(captured_b[0]), (
        f"{sentinel} from conversation A leaked into conversation B's history."
    )


@pytest.mark.integration
@pytest.mark.a2a
async def test_resume_continues_history(single_agent_hive, scripted_llm):
    """Turn 2 with reference_task_ids=[task_1_id] sees Q1+A1 in LLM messages."""
    await _assert_resume_continues_history(single_agent_hive["port"], scripted_llm)


@pytest.mark.integration
@pytest.mark.a2a
async def test_fresh_request_has_no_history(single_agent_hive, scripted_llm):
    """A new message with no reference_task_ids always starts with a clean slate.

    Even when other conversations already exist in the same task store,
    a request without reference_task_ids must see only [system, user] = 2 messages.
    This catches accidental global store leakage.
    """
    counts: list[int] = _counting_dispatch(scripted_llm)
    ctx = str(uuid4())
    client: A2AClient = _client(single_agent_hive["port"])

    _, t1 = await _send_turn(client, "Existing conversation turn one.", context_id=ctx)
    await _send_turn(
        client,
        "Existing conversation turn two.",
        context_id=ctx,
        reference_task_ids=[t1],
    )
    await _send_turn(client, "Brand new question.", context_id=str(uuid4()))
    await client.aclose()

    assert counts[2] == 2, (
        f"Fresh request should see exactly 2 messages (system+user), got {counts[2]}. "
        "History from prior conversations may be leaking."
    )


@pytest.mark.integration
@pytest.mark.a2a
async def test_context_id_alone_does_not_carry_history(single_agent_hive, scripted_llm):
    """Reusing a context_id without reference_task_ids does NOT restore history.

    context_id is a session grouping hint, not the history key.
    Only reference_task_ids drives history reconstruction.
    """
    counts: list[int] = _counting_dispatch(scripted_llm)
    ctx = str(uuid4())
    client: A2AClient = _client(single_agent_hive["port"])

    await _send_turn(client, "Turn one content.", context_id=ctx)
    await _send_turn(client, "Turn two, no reference.", context_id=ctx)
    await client.aclose()

    assert counts[1] == 2, (
        f"Second request with same context_id but no reference_task_ids should "
        f"see 2 messages (system+user), got {counts[1]}. "
        "context_id must not implicitly restore history."
    )


@pytest.mark.integration
@pytest.mark.a2a
async def test_nonexistent_reference_task_id_is_ignored(
    single_agent_hive, scripted_llm
):
    """A bogus reference_task_id is silently ignored — no crash, no phantom history.

    The BFS in HistoryRequestContextBuilder fetches None for unknown IDs and
    skips them. The request should proceed as if no references were given.
    """
    counts: list[int] = _counting_dispatch(scripted_llm)
    client: A2AClient = _client(single_agent_hive["port"])

    await _send_turn(
        client,
        "Question with bad reference.",
        context_id=str(uuid4()),
        reference_task_ids=[str(uuid4())],
    )
    await client.aclose()

    assert counts[0] == 2, (
        f"Nonexistent reference_task_id should result in 2 messages (system+user), "
        f"got {counts[0]}. Unknown IDs must be silently skipped."
    )


@pytest.mark.integration
@pytest.mark.a2a
async def test_conversations_are_isolated(single_agent_hive, scripted_llm):
    """Two separate context_ids do not bleed history into each other."""
    await _assert_conversations_are_isolated(
        single_agent_hive["port"], scripted_llm, sentinel="SENTINEL_A"
    )


@pytest.mark.integration
@pytest.mark.a2a
async def test_message_count_grows_linearly(single_agent_hive, scripted_llm):
    """Message count grows by exactly +2 per turn (no doubling).

    Turn N references only task N-1; the BFS resolves the full chain
    back to turn 1 transitively via stored reference_task_ids.

    Expected: [2, 4, 6]  — system-prompt + 1 user + (N-1)*2 history messages.
    """
    counts: list[int] = _counting_dispatch(scripted_llm)
    ctx = str(uuid4())
    client: A2AClient = _client(single_agent_hive["port"])

    _, t1 = await _send_turn(client, "Question one.", context_id=ctx)
    _, t2 = await _send_turn(
        client, "Question two.", context_id=ctx, reference_task_ids=[t1]
    )
    await _send_turn(client, "Question three.", context_id=ctx, reference_task_ids=[t2])
    await client.aclose()

    assert len(counts) == 3, f"Expected 3 LLM calls, got {len(counts)}: {counts}"
    assert counts == [2, 4, 6], (
        f"Expected [2, 4, 6] — possible doubling bug. Got: {counts}"
    )


@pytest.mark.integration
@pytest.mark.a2a
async def test_history_is_in_chronological_order(single_agent_hive, scripted_llm):
    """Messages from earlier turns appear before messages from later turns.

    The BFS in collect_all_referenced_tasks reverses the collected list to restore
    chronological order. This test catches a regression where the reversal is wrong
    or missing, causing the LLM to see history backwards.
    """
    seen_positions: dict[str, int] = {}

    async def dispatch(*, messages, **_):
        text: str = _all_text(messages)
        for marker in ("TURN_ONE", "TURN_TWO", "TURN_THREE"):
            if marker in text:
                # Record first position at which each marker appears in the flat text
                seen_positions[marker] = text.index(marker)
        return make_text_response(f"Reply {len(seen_positions)}.")

    scripted_llm.install(dispatch)

    ctx = str(uuid4())
    client: A2AClient = _client(single_agent_hive["port"])

    _, t1 = await _send_turn(client, "TURN_ONE content.", context_id=ctx)
    _, t2 = await _send_turn(
        client, "TURN_TWO content.", context_id=ctx, reference_task_ids=[t1]
    )
    await _send_turn(
        client, "TURN_THREE content.", context_id=ctx, reference_task_ids=[t2]
    )
    await client.aclose()

    assert "TURN_ONE" in seen_positions, "TURN_ONE not seen in turn-3 messages"
    assert "TURN_TWO" in seen_positions, "TURN_TWO not seen in turn-3 messages"
    assert seen_positions["TURN_ONE"] < seen_positions["TURN_TWO"], (
        "TURN_ONE appeared after TURN_TWO — history is not in chronological order"
    )


@pytest.mark.integration
@pytest.mark.a2a
async def test_shared_ancestor_task_is_not_duplicated(single_agent_hive, scripted_llm):
    """A task referenced by two paths is included in history exactly once.

    Scenario:
      - Turn 1 (task_A): "ANCESTOR content"
      - Turn 2a (task_B): references task_A
      - Turn 2b (task_C): also references task_A (independent branch)
      - Turn 3: references both task_B and task_C explicitly

    The BFS visited-set must deduplicate task_A. Without it, ANCESTOR appears twice,
    inflating the message count and confusing the LLM.
    """
    counts: list[int] = _counting_dispatch(scripted_llm)
    ctx = str(uuid4())
    client: A2AClient = _client(single_agent_hive["port"])

    # Common ancestor
    _, task_a = await _send_turn(client, "ANCESTOR content.", context_id=ctx)

    # Two independent turns that both reference the ancestor
    _, task_b = await _send_turn(
        client, "Branch B.", context_id=ctx, reference_task_ids=[task_a]
    )
    _, task_c = await _send_turn(
        client, "Branch C.", context_id=ctx, reference_task_ids=[task_a]
    )

    # Turn that explicitly references both branches
    await _send_turn(
        client, "Merge turn.", context_id=ctx, reference_task_ids=[task_b, task_c]
    )
    await client.aclose()

    # If task_A were duplicated, turn 4 would see: sys + A + A + B + C + merge_q = 6+
    # Correctly deduplicated: sys + A + B + C + merge_q = 5... wait, let me think:
    # BFS from [task_b, task_c]:
    #   fetch task_b → history=[B_user, B_asst], refs=[task_a]
    #   fetch task_c → history=[C_user, C_asst], refs=[task_a]
    #   fetch task_a → history=[A_user, A_asst], refs=[]  (visited once)
    # reversed order: task_a, task_b, task_c
    # history = A_user + A_asst + B_user + B_asst + C_user + C_asst + merge_q
    # + system = 8 messages
    assert counts[3] == 8, (
        f"Turn 4 (merge): expected 8 messages (sys + 3×2 history + merge_q), "
        f"got {counts[3]}. "
        "If > 8: ancestor task is likely duplicated in the history."
    )


@pytest.mark.integration
@pytest.mark.a2a
async def test_explicit_multi_reference_merges_two_chains(
    single_agent_hive, scripted_llm
):
    """Passing two task IDs in reference_task_ids directly merges both chains.

    Unlike the typical linear-chain pattern (each turn refs only the previous),
    this tests passing multiple IDs at once — the client controls the merge
    explicitly rather than relying on BFS transitivity.
    """
    captured = _capturing_dispatch(scripted_llm)
    ctx = str(uuid4())
    client: A2AClient = _client(single_agent_hive["port"])

    # Two independent single-turn conversations
    _, task_x = await _send_turn(client, "CHAIN_X question.", context_id=ctx)
    _, task_y = await _send_turn(client, "CHAIN_Y question.", context_id=ctx)

    # Merge both explicitly
    await _send_turn(
        client, "Merge question.", context_id=ctx, reference_task_ids=[task_x, task_y]
    )
    await client.aclose()

    assert len(captured) == 3
    merge_text = _all_text(captured[3])
    assert "CHAIN_X question." in merge_text, "CHAIN_X not present in merged history"
    assert "CHAIN_Y question." in merge_text, "CHAIN_Y not present in merged history"


@pytest.mark.integration
@pytest.mark.a2a
@pytest.mark.postgres
async def test_postgres_resume_continues_history(pg_hive, scripted_llm):
    """PostgreSQL variant of test_resume_continues_history."""
    await _assert_resume_continues_history(pg_hive["port"], scripted_llm)


@pytest.mark.integration
@pytest.mark.a2a
@pytest.mark.postgres
async def test_postgres_conversations_are_isolated(pg_hive, scripted_llm):
    """PostgreSQL variant of test_conversations_are_isolated."""
    await _assert_conversations_are_isolated(
        pg_hive["port"], scripted_llm, sentinel="PG_SENTINEL_A"
    )


@pytest.mark.integration
@pytest.mark.a2a
@pytest.mark.postgres
async def test_postgres_history_survives_server_restart(
    postgres_url: str, scripted_llm
):
    """History persists across a full server restart when using DatabaseTaskStore.

    Phase 1: start server → send turn 1 → stop server entirely.
    Phase 2: start a NEW server with the same DB → send turn 2 → verify
             turn-1 content appears in the LLM's message list.
    """
    pg_url = postgres_url

    # ── Phase 1: first server ──────────────────────────────────────────────
    async def dispatch_phase1(*, messages, **_):
        return make_text_response("Phase-1 answer.")

    scripted_llm.install(dispatch_phase1)

    colony1, _, sock1 = _make_colony(db_url=pg_url)
    port1 = sock1.getsockname()[1]
    server1, srv_task1 = await _start_server(colony1.asgi(agent_name="sticky"), sock1)

    ctx = str(uuid4())
    client1: A2AClient = _client(port1)
    _, task_1_id = await _send_turn(client1, "Persistent question?", context_id=ctx)
    await client1.aclose()
    await _stop_server(server1, srv_task1)  # all in-memory state is gone
    await colony1.aclose()
    await _drain_loop()  # flush asyncpg cleanup in this event loop

    # ── Phase 2: brand-new server with the same DB ─────────────────────────
    phase2_messages: list[list] = []

    async def dispatch_phase2(*, messages, **_):
        phase2_messages.append(list(messages))
        return make_text_response("Phase-2 answer.")

    scripted_llm.install(dispatch_phase2)

    colony2, _, sock2 = _make_colony(db_url=pg_url)
    port2 = sock2.getsockname()[1]
    server2, srv_task2 = await _start_server(colony2.asgi(agent_name="sticky"), sock2)

    client2: A2AClient = _client(port2)
    await _send_turn(
        client2,
        "What did I ask before?",
        context_id=ctx,
        reference_task_ids=[task_1_id],
    )
    await client2.aclose()
    await _stop_server(server2, srv_task2)
    await colony2.aclose()
    await _drain_loop()  # flush asyncpg cleanup in this event loop

    # ── Assert ─────────────────────────────────────────────────────────────
    assert phase2_messages, "Phase-2 dispatch was never called"
    assert "Persistent question?" in _all_text(phase2_messages[0]), (
        "Turn-1 question not found after server restart — "
        "DatabaseTaskStore did not persist the task.\n"
        f"Phase-2 messages: {phase2_messages[0]}"
    )
