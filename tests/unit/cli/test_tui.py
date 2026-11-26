from __future__ import annotations

import asyncio
import json
import re
import sys
import time
from pathlib import Path

import pytest
from textual.widgets import Collapsible, DataTable, Input, Markdown, Static, Tree

from ant_ai.cli.tui import ANT, SIDE_FROM, AntApp, PermissionScreen

AGENT = Path(__file__).with_name("scripted_acp_agent.py")


async def _ready(app: AntApp) -> None:
    """Wait for the agent to start, failing with its stderr if it does not."""
    try:
        await asyncio.wait_for(app._connected.wait(), 10)
    except TimeoutError:
        pytest.fail("the agent did not start:\n" + "\n".join(app.session.stderr))


async def _run_turn(app: AntApp, pilot, answer: str) -> None:
    await _ready(app)
    await pilot.click("#ask")
    await pilot.press(*"create a.txt", "enter")
    for _ in range(100):
        if isinstance(app.screen, PermissionScreen):
            break
        await pilot.pause(0.05)
    assert isinstance(app.screen, PermissionScreen)
    await pilot.press(answer)
    for _ in range(200):
        if app._running_turn is not None and app._running_turn.is_finished:
            break
        await pilot.pause(0.05)
    await pilot.pause()


@pytest.mark.unit
async def test_a_turn_fills_the_conversation_and_the_panel(tmp_path):
    app = AntApp(
        [sys.executable, str(AGENT), str(tmp_path)],
        tmp_path,
        label="scripted",
        max_steps=5,
    )
    async with app.run_test(size=(140, 40)) as pilot:
        await _run_turn(app, pilot, "y")

        (call,) = app.query(Collapsible)
        assert "write_file" in str(call.title) and "written" in str(call.title)
        assert call.collapsed
        assert "Wrote a.txt." in list(app.query(Markdown))[-1].source
        # The user's task and the answer are also listed in the panel, with the time.
        kinds = [
            (sender, to, what) for sender, to, what, _, _ in app.view.messages.values()
        ]
        assert kinds == [("you", "ant", "task"), ("ant", "you", "answer")]
        assert app.query_one("#panel").display
        assert "context" in str(app.query_one("#status", Static).render())
        agents = app.query_one("#agents", Tree)
        assert "waiting" in str(agents.root.children[0].label)
        assert str(call.title).endswith("s")
        status = str(app.query_one("#status", Static).render())
        assert "2 steps" in status and status.endswith("1 tool")
    assert (tmp_path / "a.txt").read_text() == "x"


@pytest.mark.unit
async def test_a_refusal_shows_as_a_failed_call(tmp_path):
    app = AntApp(
        [sys.executable, str(AGENT), str(tmp_path)], tmp_path, label="scripted"
    )
    async with app.run_test(size=(140, 40)) as pilot:
        await _run_turn(app, pilot, "n")

        (call,) = app.query(Collapsible)
        assert "✗ The user did not allow" in str(call.title)
        assert call.has_class("error")
        assert "1 tool ✗1" in str(app.query_one("#status", Static).render())
    assert not (tmp_path / "a.txt").exists()


@pytest.mark.unit
async def test_a_result_right_after_its_call_is_shown(tmp_path):
    from ant_ai.core.events import ToolCallingEvent, ToolResultEvent
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["true"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        app.view.reset_agents("ant")
        call = ToolCall(
            id="c1",
            function=ToolFunction(name="read_file", arguments='{"path": "a.py"}'),
        )
        app.view.event(ToolCallingEvent(content="Reading.", tool_calls=(call,)))
        app.view.event(
            ToolResultEvent(tool_call_id="c1", name="read_file", content="x = 1")
        )
        await pilot.pause()

        (box,) = app.query(Collapsible)
        assert "1 line" in str(box.title) or "x = 1" in str(box.title)


@pytest.mark.unit
def test_a_command_result_reads_like_the_terminal():
    from ant_ai.cli.tui import _readable

    shown = _readable('{"stdout": "3 passed\\n", "stderr": "", "returncode": 0}')
    assert shown == "3 passed\n(exit 0)"
    assert _readable("plain text") == "plain text"


@pytest.mark.unit
async def test_text_shows_while_it_is_written_and_the_ant_walks(tmp_path):
    from ant_ai.core.events import FinalAnswerEvent

    app = AntApp(["true"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        app.view.reset_agents("ant")
        with app.view.turn():
            app.view.partial("Hel")
            app.view.partial("lo")
            await pilot.pause(0.3)
            (answer,) = app.query(Markdown)
            assert answer.source == "Hello"
            status = str(app.query_one("#status", Static).render())
            # The ant leads the status line while the agent works.
            assert any(status.startswith(f"{ant} writing") for ant in ANT)
            app.view.event(FinalAnswerEvent(content="Hello"))
            await pilot.pause()
            # The same widget: the answer is not removed and drawn again.
            assert list(app.query(Markdown)) == [answer]
        assert str(app.query_one("#status", Static).render()).startswith(
            f"{ANT[0]} ready"
        )


@pytest.mark.unit
def test_a_huge_one_line_result_is_cut_and_a_list_reads_one_item_per_line():
    from ant_ai.cli.tui import RESULT_CHARS, _shown

    huge = json.dumps([f"dir/file_{i}.py" for i in range(200_000)])
    shown = _shown(huge)
    assert len(shown) <= RESULT_CHARS + 100
    assert shown.startswith("dir/file_0.py\ndir/file_1.py\n")
    assert "more" in shown.splitlines()[-1]


@pytest.mark.unit
async def test_the_timer_does_not_fail_once_the_app_has_closed(tmp_path):
    app = AntApp(["true"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)):
        app.view.doing("thinking")
    # Closing mid-turn: the status timer can still fire once, with no widgets left.
    app.view.tick()


async def _wait_for(condition, pilot, tries: int = 200) -> None:
    for _ in range(tries):
        if condition():
            return
        await pilot.pause(0.05)
    raise AssertionError("timed out")


def _said(app: AntApp) -> list[str]:
    return [str(s.render()) for s in app.query(".user")]


@pytest.mark.unit
async def test_a_message_sent_during_a_turn_waits_and_runs_after_it(tmp_path):
    app = AntApp(
        [sys.executable, str(AGENT), str(tmp_path), "slow"], tmp_path, label="scripted"
    )
    async with app.run_test(size=(140, 40)) as pilot:
        await _ready(app)
        await pilot.click("#ask")
        await pilot.press(*"one", "enter")
        await _wait_for((tmp_path / "started").exists, pilot)

        await pilot.press(*"two", "enter")
        await pilot.pause()
        assert app.query_one("#ask").value == ""
        assert _said(app) == ["› one"]
        assert any("queued" in str(n.render()) for n in app.query(".note"))

        (tmp_path / "go").touch()
        await _wait_for(lambda: len(app.query(Markdown)) == 2, pilot)
        assert _said(app) == ["› one", "› two"]
        assert [m.source for m in app.query(Markdown)] == ["first", "second"]


@pytest.mark.unit
async def test_stopping_a_turn_gives_the_queued_message_back(tmp_path):
    app = AntApp(
        [sys.executable, str(AGENT), str(tmp_path), "stuck"], tmp_path, label="scripted"
    )
    async with app.run_test(size=(140, 40)) as pilot:
        await _ready(app)
        await pilot.click("#ask")
        await pilot.press(*"one", "enter")
        await _wait_for((tmp_path / "started").exists, pilot)
        await pilot.press(*"two", "enter")

        await pilot.press("escape")
        await _wait_for(lambda: app._running_turn.is_finished, pilot)
        await pilot.pause()
        assert app.query_one("#ask").value == "two"
        assert _said(app) == ["› one"]


@pytest.mark.unit
async def test_new_text_right_after_a_step_does_not_clash_with_the_old_one(tmp_path):
    from ant_ai.core.events import ToolCallingEvent
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["true"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        app.view.reset_agents("ant")
        call = ToolCall(id="c1", function=ToolFunction(name="list_dir", arguments="{}"))
        with app.view.turn():
            app.view.partial("I list the files.")
            app.view.event(ToolCallingEvent(content="I list.", tool_calls=(call,)))
            # The old live text is still being removed when the next one starts.
            app.view.partial("Now I read one.")
            await pilot.pause()
            assert [m.source for m in app.query(Markdown)] == [
                "I list the files.",
                "Now I read one.",
            ]


@pytest.mark.unit
async def test_a_call_with_only_a_message_shows_as_a_sub_agent(tmp_path):
    from ant_ai.core.events import ToolCallingEvent
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["true"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        app.view.reset_agents("ant")
        # A local agent of `as_tool()` and an A2A agent both take one `message`.
        calls = (
            ToolCall(
                id="c1",
                function=ToolFunction(
                    name="reviewer", arguments='{"message": "check a.py"}'
                ),
            ),
            ToolCall(
                id="c2", function=ToolFunction(name="todo", arguments='{"task": "x"}')
            ),
        )
        app.view.event(ToolCallingEvent(content="", tool_calls=calls))
        await pilot.pause()

        root = app.query_one("#agents", Tree).root.children[0]
        assert [n.data for n in root.children] == ["reviewer"]
        assert "A2A" not in str(root.children[0].label)
        used = str(app.query_one("#used", Static).render())
        assert "todo" in used and "reviewer" not in used


@pytest.mark.unit
async def test_the_plan_the_step_limit_and_a_question_are_shown(tmp_path):
    from ant_ai.cli.render import PLAN_TOOL
    from ant_ai.core.events import (
        ClarificationNeededEvent,
        MaxStepsReachedEvent,
        ToolCallingEvent,
        ToolResultEvent,
    )
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["true"], tmp_path, label="scripted", context_size=200_000)
    async with app.run_test(size=(140, 40)) as pilot:
        app.view.reset_agents("ant")
        todos = [
            {"content": "read the code", "status": "completed"},
            {"content": "fix the bug", "status": "in_progress"},
        ]
        call = ToolCall(
            id="p1",
            function=ToolFunction(
                name=PLAN_TOOL, arguments=json.dumps({"todos": todos})
            ),
        )
        app.view.event(ToolCallingEvent(content="", tool_calls=(call,)))
        app.view.event(ToolResultEvent(tool_call_id="p1", name=PLAN_TOOL, content="ok"))
        app.view.event(MaxStepsReachedEvent())
        app.view.event(ClarificationNeededEvent(content="Which file?"))
        await pilot.pause()

        plan = str(app.query_one(".plan", Static).render())
        assert "read the code" in plan and "fix the bug" in plan
        assert "plan 1/2" in plan and not app.query(Collapsible)
        notes = [str(n.render()) for n in app.query(".note")]
        assert "stopped: step limit reached" in notes
        assert any("Which file?" in str(s.render()) for s in app.query(Static))
        assert "/200k" in str(app.query_one("#status", Static).render())


@pytest.mark.unit
async def test_keys_open_the_tools_and_start_a_new_conversation(tmp_path):
    app = AntApp(
        [sys.executable, str(AGENT), str(tmp_path)], tmp_path, label="scripted"
    )
    async with app.run_test(size=(140, 40)) as pilot:
        await _run_turn(app, pilot, "y")

        await pilot.press("ctrl+o")
        assert not app.query_one(Collapsible).collapsed
        await pilot.press("ctrl+o")
        assert app.query_one(Collapsible).collapsed

        await pilot.press("ctrl+l")
        await pilot.pause()
        assert not app.query(Collapsible)
        assert [str(n.render()) for n in app.query(".note")] == ["new conversation"]


@pytest.mark.unit
def test_the_context_size_is_unknown_without_a_known_model():
    from ant_ai.cli.tui import context_size

    assert context_size(None) is None
    assert context_size("no-such-provider/no-such-model") is None


@pytest.mark.unit
async def test_a_new_conversation_clears_every_panel(tmp_path):
    app = AntApp(
        [sys.executable, str(AGENT), str(tmp_path)],
        tmp_path,
        label="scripted",
        max_steps=5,
    )
    async with app.run_test(size=(140, 40)) as pilot:
        await _run_turn(app, pilot, "y")
        app.view._message("ant", "reviewer", "task", "check a.txt")

        await pilot.press("ctrl+l")
        await pilot.pause()

        assert app.query_one("#messages", DataTable).row_count == 0
        assert app.query_one("#panel").display
        assert "none yet" in str(app.query_one("#used", Static).render())
        status = str(app.query_one("#status", Static).render()).split()
        assert status == [ANT[0], "ready", "context", "≈0.0k"]


@pytest.mark.unit
async def test_a_stopped_turn_closes_the_calls_still_running(tmp_path):
    from ant_ai.core.events import ToolCallingEvent
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["true"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        app.view.reset_agents("ant")
        calls = (
            ToolCall(id="c1", function=ToolFunction(name="run", arguments="{}")),
            ToolCall(
                id="c2",
                function=ToolFunction(name="reviewer", arguments='{"message": "x"}'),
            ),
        )
        with app.view.turn():
            app.view.event(ToolCallingEvent(content="", tool_calls=calls))
        await pilot.pause()

        titles = [str(c.title) for c in app.query(Collapsible)]
        assert all(t.endswith("stopped") for t in titles), titles
        sub = app.query_one("#agents", Tree).root.children[0].children[0]
        assert "stopped" in str(sub.label)
        assert app.view._calls == {}


@pytest.mark.unit
async def test_an_agent_that_exits_mid_turn_is_reported(tmp_path):
    app = AntApp(
        [sys.executable, str(AGENT), str(tmp_path), "crash"], tmp_path, label="s"
    )
    async with app.run_test(size=(140, 40)) as pilot:
        await _ready(app)
        await pilot.click("#ask")
        await pilot.press(*"one", "enter")
        await _wait_for(lambda: app._running_turn.is_finished, pilot)
        await pilot.pause()

        notes = [str(n.render()) for n in app.query(".note")]
        assert any(n.startswith("the agent stopped") for n in notes), notes
        assert app.is_running

        await pilot.press(*"two", "enter")
        await pilot.pause()
        notes = [str(n.render()) for n in app.query(".note")]
        assert notes[-1] == "the agent is not running: restart ant"


@pytest.mark.unit
async def test_a_message_can_be_read_in_full(tmp_path):
    from ant_ai.cli.tui import MessageScreen

    app = AntApp(["true"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        app.view.reset_agents("ant")
        long = "Read src/app.py and tell me " + "why it fails " * 20
        app.view._message("ant", "reviewer", "task", long)
        await pilot.pause()

        table = app.query_one("#messages", DataTable)
        table.focus()
        await pilot.press("enter")
        await pilot.pause()

        assert isinstance(app.screen, MessageScreen)
        assert long in str(app.screen.query_one("#message-text", Static).render())
        await pilot.press("escape")
        assert not isinstance(app.screen, MessageScreen)


@pytest.mark.unit
async def test_the_input_takes_keys_before_the_agent_is_connected(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.pause()
        assert not app._connected.is_set()
        assert app.focused is app.query_one("#ask")


@pytest.mark.unit
async def test_the_permission_question_reads_without_the_terminal_keys(tmp_path):
    app = AntApp(["true"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        app.push_screen(PermissionScreen("allow run `pytest`? y/n/a"))
        await pilot.pause()
        label = str(app.screen.query_one("Label").render())
        assert label == "allow run `pytest`?"


def test_a_command_summary_shows_the_last_line_of_its_output():
    from ant_ai.cli.render import result_summary

    out = json.dumps(
        {"stdout": "..F\n6 failed, 3 passed\n", "stderr": "warn\n", "returncode": 1}
    )
    assert result_summary("run_command", out) == ("exit 1 · 6 failed, 3 passed", False)
    only_err = json.dumps({"stdout": "", "stderr": "not found\n", "returncode": 2})
    assert result_summary("run_command", only_err) == ("exit 2 · not found", False)


@pytest.mark.unit
async def test_the_reasoning_of_the_model_is_kept_folded(tmp_path):
    from ant_ai.core.events import ReasoningEvent

    app = AntApp(["true"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        app.view.reset_agents("ant")
        app.view.event(ReasoningEvent(content="The test expects 2, "))
        app.view.event(ReasoningEvent(content="so the + 1 is wrong."))
        await pilot.pause()

        (thought,) = app.query(Collapsible)
        assert thought.collapsed and "thinking" in str(thought.title)
        body = str(thought.query_one(".result", Static).render())
        assert body == "The test expects 2, so the + 1 is wrong."


@pytest.mark.unit
async def test_a_task_given_on_the_command_line_is_sent_once_the_agent_starts(
    tmp_path,
):
    app = AntApp(
        [sys.executable, str(AGENT), str(tmp_path)],
        tmp_path,
        label="scripted",
        task="create a.txt",
    )
    async with app.run_test(size=(140, 40)) as pilot:
        await _wait_for(lambda: isinstance(app.screen, PermissionScreen), pilot)
        await pilot.press("y")
        await _wait_for(lambda: len(app.query(Markdown)) == 2, pilot)
        assert _said(app) == ["› create a.txt"]
    assert (tmp_path / "a.txt").read_text() == "x"


@pytest.mark.unit
async def test_a_message_sent_before_the_agent_starts_waits_for_it(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        await pilot.press(*"hello", "enter")
        await pilot.pause()
        assert app._queued == ["hello"] and app._running_turn is None
        assert any("queued: hello" in str(n.render()) for n in app.query(".note"))


@pytest.mark.unit
async def test_a_withdrawn_question_closes_its_dialog(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 40)) as pilot:
        question = asyncio.ensure_future(app.ask_permission("allow run `ls`?"))
        await _wait_for(lambda: isinstance(app.screen, PermissionScreen), pilot)
        question.cancel()
        await _wait_for(lambda: not isinstance(app.screen, PermissionScreen), pilot)
        with pytest.raises(asyncio.CancelledError):
            await question


@pytest.mark.unit
async def test_stopping_puts_the_queued_message_before_the_draft(tmp_path):
    app = AntApp(
        [sys.executable, str(AGENT), str(tmp_path), "stuck"], tmp_path, label="scripted"
    )
    async with app.run_test(size=(140, 40)) as pilot:
        await _ready(app)
        await pilot.click("#ask")
        await pilot.press(*"one", "enter")
        await _wait_for((tmp_path / "started").exists, pilot)
        await pilot.press(*"two", "enter", *"three")

        await pilot.press("escape")
        await _wait_for(lambda: app._running_turn.is_finished, pilot)
        await pilot.pause()
        assert app.query_one("#ask").value == "two three"


@pytest.mark.unit
@pytest.mark.parametrize("width", [80, 100, 150])
async def test_the_panel_fills_the_width_it_is_given(tmp_path, width):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(width, 30)) as pilot:
        await pilot.pause()
        side = width >= SIDE_FROM
        assert app.query_one("#panel").display is side
        if not side:
            for selector in ("#ask", "#conversation", "#status"):
                assert app.query_one(selector).region.right == width - 1
        app.view._message("ant", "reviewer", "task", "check a.py")
        if not side:
            await pilot.press("ctrl+t")
        await pilot.pause()
        panel = app.query_one("#panel").region
        assert panel.right == width - 1
        if width < SIDE_FROM:
            assert panel.height <= 6
        else:
            assert panel.bottom == app.query_one("#ask").region.y


@pytest.mark.unit
async def test_the_panel_moves_beside_the_conversation_on_a_wide_screen(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 30)) as pilot:
        app.view._message("ant", "reviewer", "task", "check a.py and say what is wrong")
        await pilot.pause()
        panel = app.query_one("#panel").region
        chat = app.query_one("#conversation").region
        assert panel.x == chat.right and panel.y == chat.y
        who, text = (
            app.query_one("#messages", DataTable).get_row_at(0)[0].plain.split("\n")
        )
        assert who.startswith("ant → reviewer")
        assert re.search(r"task \d\d:\d\d$", who)
        assert text.startswith("check a.py")
        await pilot.resize_terminal(100, 30)
        await pilot.press("ctrl+t")
        await pilot.pause()
        panel = app.query_one("#panel").region
        chat = app.query_one("#conversation").region
        assert panel.bottom <= chat.y and chat.width == 98
        assert app.query_one("#messages", DataTable).row_count == 1


@pytest.mark.unit
async def test_the_plan_never_changes_the_height_of_the_conversation(tmp_path):
    from ant_ai.cli.render import PLAN_TOOL
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        box = app.query_one("#conversation")
        height = box.region.height
        for status in ("pending", "completed"):
            todos = [{"content": "fix the bug", "status": status}]
            app.view._plan(
                ToolCall(
                    id="p1",
                    function=ToolFunction(
                        name=PLAN_TOOL, arguments=json.dumps({"todos": todos})
                    ),
                )
            )
            await pilot.pause()
            assert box.region.height == height
            assert len(app.query(".plan")) == 1


@pytest.mark.unit
async def test_the_step_shows_while_thinking_and_stays_right(tmp_path):
    from ant_ai.core.events import ToolCallingEvent, ToolResultEvent
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(100, 30)) as pilot:
        app.view.reset_agents("ant")
        calls = tuple(
            ToolCall(id=i, function=ToolFunction(name="list_dir", arguments="{}"))
            for i in ("a", "b")
        )

        def status() -> str:
            return str(app.query_one("#status", Static).render())

        with app.view.turn():
            assert "step 1" in status()
            app.view.event(ToolCallingEvent(content="", tool_calls=calls))
            assert "step 1" in status()
            for n, step in (("a", "step 1"), ("b", "step 2")):
                app.view.event(
                    ToolResultEvent(tool_call_id=n, name="list_dir", content="[]")
                )
                assert step in status()
            app.view.partial("Half an ans")
        await pilot.pause()
        assert app.query_one(Markdown).source == "Half an ans"
        assert "2 steps" in status()


@pytest.mark.unit
async def test_the_panel_opens_and_closes_only_with_its_key(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(80, 24)) as pilot:
        await pilot.pause()
        panel = app.query_one("#panel")
        chat = app.query_one("#conversation")
        height = chat.region.height
        app.view._message("ant", "reviewer", "task", "check a.py")
        await pilot.pause()
        assert not panel.display and chat.region.height == height

        await pilot.press("h", "i", "ctrl+t")
        await pilot.pause()
        assert panel.display
        # The messages take half, the agents and the tools a quarter each.
        for selector in ("#agents", "#tools"):
            assert app.query_one(selector).region.width >= 18
        assert app.query_one("#messages").region.width >= 36
        # Closing it with the focus inside gives the keys back to the input.
        app.query_one("#messages").focus()
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert not panel.display and chat.region.height == height
        assert app.focused is app.query_one("#ask")
        assert app.query_one("#ask", Input).value == "hi"


@pytest.mark.unit
async def test_the_panel_counts_the_tools_of_the_whole_conversation(tmp_path):
    from ant_ai.core.events import ToolCallingEvent, ToolResultEvent
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(100, 30)) as pilot:
        app.view.reset_agents("ant")
        for n, failed in (("a", False), ("b", True)):
            call = ToolCall(
                id=n, function=ToolFunction(name="list_dir", arguments="{}")
            )
            with app.view.turn():
                app.view.event(ToolCallingEvent(content="", tool_calls=(call,)))
                app.view.event(
                    ToolResultEvent(
                        tool_call_id=n, name="list_dir", content="[]", is_error=failed
                    )
                )
        await pilot.press("ctrl+t")
        await pilot.pause()
        used = str(app.query_one("#used", Static).render())
        assert used.split() == ["list_dir", "2", "✗1"]


@pytest.mark.unit
async def test_the_panel_toggles_during_a_permission_question(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.pause()
        app.push_screen(PermissionScreen("allow write a.py?"))
        await pilot.pause()
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert app.screen_stack[0].query_one("#panel").display
        assert isinstance(app.screen, PermissionScreen)


@pytest.mark.unit
async def test_closing_the_panel_leaves_a_short_conversation_at_the_top(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(100, 30)) as pilot:
        app.view.user("hello")
        box = app.query_one("#conversation")
        # Just short enough to fit while the panel is closed.
        while not box.max_scroll_y:
            app.view.note("line")
            await pilot.pause(0.1)
        app.query(".note").last().remove()
        await pilot.pause(0.3)
        assert box.max_scroll_y == 0
        first = app.query_one(".user")
        top = first.region.y
        await pilot.press("ctrl+t")
        await pilot.pause(0.3)
        assert box.max_scroll_y > 0 and box.scroll_y >= box.max_scroll_y - 1
        await pilot.press("ctrl+t")
        await pilot.pause(0.3)
        assert first.region.y == top


@pytest.mark.unit
async def test_what_is_written_does_not_move_until_the_screen_is_full(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(100, 30)) as pilot:
        app.view.user("hello")
        await pilot.pause(0.3)
        first = app.query_one(".user")
        top = first.region.y
        assert top <= app.query_one("#conversation").region.y + 1
        app.view.note("one more line")
        await pilot.pause(0.3)
        assert first.region.y == top
        for n in range(40):
            app.view.note(f"line {n}")
        await pilot.pause(0.5)
        box = app.query_one("#conversation")
        assert box.max_scroll_y > 0 and box.scroll_y >= box.max_scroll_y - 1


@pytest.mark.unit
async def test_the_status_line_keeps_the_plan_in_sight(tmp_path):
    from ant_ai.cli.render import PLAN_TOOL
    from ant_ai.core.events import FinalAnswerEvent
    from ant_ai.core.message import ToolCall, ToolFunction

    def plan(*statuses: str) -> ToolCall:
        todos = [{"content": f"item {i}", "status": s} for i, s in enumerate(statuses)]
        return ToolCall(
            id="p1",
            function=ToolFunction(
                name=PLAN_TOOL, arguments=json.dumps({"todos": todos})
            ),
        )

    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(80, 24)) as pilot:
        status = app.query_one("#status", Static)
        with app.view.turn():
            app.view.show_status()
            assert "plan" not in str(status.render())
            app.view._plan(plan("completed", "in_progress", "pending"))
            app.view.show_status()
            assert "ready" not in str(status.render())
            assert "plan 1/3 · step" in str(status.render())
            app.view.event(FinalAnswerEvent(content="done"))
        await pilot.pause()
        assert "plan 1/3" in str(status.render())
        with app.view.turn():
            app.view.show_status()
            assert "plan" not in str(status.render())


@pytest.mark.unit
async def test_a_busy_status_line_fits_80_columns(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(80, 24)):
        view = app.view
        view._doing, view._plan_done = "run_command", "plan 2/3"
        view.steps, view.tool_calls, view._started = 16, 12, time.monotonic() - 12.3
        view.show_status()
        status = app.query_one("#status", Static)
        line = str(status.render()).rstrip()
        assert line.endswith("12 tools") and len(line) <= status.region.width


@pytest.mark.unit
async def test_the_panel_fits_what_it_shows_and_only_grows_while_open(tmp_path):
    from ant_ai.cli.tui import ToolStats

    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(100, 30)) as pilot:
        view, panel = app.view, app.query_one("#panel")
        view.reset_agents("ant")

        async def rows() -> int:
            await pilot.pause()
            return panel.region.height - 2

        await pilot.press("ctrl+t")
        assert await rows() == 1
        for name in ("read_file", "search", "edit_file"):
            view._used[name] = ToolStats(calls=1)
        view._show_used()
        assert await rows() == 3
        for i in range(20):
            view._message("ant", "reviewer", "task", f"check {i}")
        assert await rows() == 30 // 4 - 2
        await pilot.press("ctrl+t", "ctrl+t")
        assert await rows() == 30 // 4 - 2
        view.reset("ant")
        assert await rows() == 1


@pytest.mark.unit
async def test_a_new_conversation_starts_at_the_top(tmp_path):
    from ant_ai.core.events import (
        FinalAnswerEvent,
        ToolCallingEvent,
        ToolResultEvent,
    )
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(80, 24)) as pilot:
        view, box = app.view, app.query_one("#conversation")
        view.user("task")
        with view.turn():
            for n in range(20):
                call = ToolCall(
                    id=str(n), function=ToolFunction(name="read_file", arguments="{}")
                )
                view.event(ToolCallingEvent(content="", tool_calls=(call,)))
                view.event(
                    ToolResultEvent(
                        tool_call_id=call.id, name="read_file", content="ok"
                    )
                )
            view.event(FinalAnswerEvent(content="done"))
        await pilot.pause(0.6)
        assert box.max_scroll_y > 0
        await box.remove_children()
        view.reset("ant")
        view.user("again")
        await pilot.pause(0.6)
        assert box.scroll_y == 0
        assert app.query_one(".user").region.y <= box.region.y + 1


@pytest.mark.unit
async def test_the_side_column_shows_the_plan_and_the_context(tmp_path):
    from ant_ai.cli.render import PLAN_TOOL
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["sleep", "5"], tmp_path, label="scripted", context_size=200_000)
    async with app.run_test(size=(140, 30)) as pilot:
        await pilot.pause()
        plan = app.query_one("#side-plan", Static)
        status = app.query_one("#status", Static)
        panel = app.query_one("#panel")
        assert plan.display and "no plan yet" in str(plan.render())
        # Under the agents, touching them.
        assert plan.region.y == app.query_one("#agents").region.bottom
        # The column reaches the input; the context is at the left, after the state.
        assert panel.region.bottom == app.query_one("#ask").region.y
        assert "<1%" in str(status.render())
        at = str(status.render()).index("context")

        todos = [{"content": "Read a.py", "status": "in_progress"}]
        args = json.dumps({"todos": todos})
        app.view._plan(
            ToolCall(id="1", function=ToolFunction(name=PLAN_TOOL, arguments=args))
        )
        await pilot.pause()
        assert plan.border_title == "PLAN 0/1" and "Read a.py" in str(plan.render())
        in_chat = app.query_one("#conversation .plan", Static)
        assert not in_chat.display
        assert "plan" not in str(app.query_one("#status", Static).render())

        await pilot.press("ctrl+t")
        await pilot.pause()
        assert not panel.display
        assert app.query_one("#conversation").region.right == 139
        assert in_chat.display
        assert "plan 0/1" in str(app.query_one("#status", Static).render())
        # The context does not move when the column goes away.
        assert str(status.render()).index("context") == at

        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert not plan.display and "<1%" in str(status.render())


@pytest.mark.unit
async def test_the_status_says_foraging_while_the_model_thinks(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 30)):
        app.view.doing("thinking")
        assert " foraging" in str(app.query_one("#status", Static).render())
        app.view.doing("run_command")
        assert " run_command" in str(app.query_one("#status", Static).render())


def _rows(app):
    table = app.query_one("#messages", DataTable)
    lines = []
    for index in range(table.row_count):
        lines += table.get_row_at(index)[0].plain.split("\n")
    return lines, table.scrollable_content_region.width


@pytest.mark.unit
async def test_message_rows_follow_the_width_across_a_resize(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    long = "check a.py and say what is wrong with it " * 3
    async with app.run_test(size=(100, 30)) as pilot:
        await pilot.press("ctrl+t")
        app.view._message("ant", "reviewer", "task", long)
        for width in (140, 100, 140, 119, 120):
            await pilot.resize_terminal(width, 30)
            await pilot.pause()
            await pilot.pause()
            lines, room = _rows(app)
            assert all(len(line) <= room for line in lines), (width, lines, room)
            assert lines[-1].endswith("…")
        assert re.search(r"task \d\d:\d\d$", lines[0])


@pytest.mark.unit
async def test_a_message_that_came_while_the_panel_was_shut_fits_when_opened(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(80, 24)) as pilot:
        app.view._message("ant", "reviewer", "task", "check a.py " * 20)
        await pilot.pause()
        await pilot.press("ctrl+t")
        await pilot.pause()
        await pilot.pause()
        lines, room = _rows(app)
        assert room and len(lines[0]) <= room and lines[0].endswith("…")
        # Both names, the receiver too, on a narrow screen.
        assert re.match(r"\d\d:\d\d ant→reviewer", lines[0])


@pytest.mark.unit
async def test_the_scrollbar_does_not_cut_the_message_rows(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 30)) as pilot:
        for number in range(12):
            app.view._message("ant", "reviewer", "answer", f"{number} " + "x" * 80)
        await pilot.pause()
        await pilot.pause()
        table = app.query_one("#messages", DataTable)
        assert table.max_scroll_y > 0
        lines, room = _rows(app)
        assert room < table.content_region.width + 1
        assert all(len(line) <= room for line in lines)


@pytest.mark.unit
@pytest.mark.parametrize("width", [100, 140])
async def test_the_counters_do_not_move_when_the_context_fills(tmp_path, width):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted", context_size=200_000)
    async with app.run_test(size=(width, 30)) as pilot:
        await pilot.pause()
        app.view.steps = 1
        places = []
        for tokens in (0, 8_000, 150_000, 200_000, 2_000_000):
            app.view._context_chars = 0
            app.view._seen("x" * tokens * 4)
            places.append(str(app.query_one("#status", Static).render()).index("step"))
        assert len(set(places)) == 1, places


@pytest.mark.unit
async def test_the_panel_stays_hidden_after_a_resize_once_hidden(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(100, 30)) as pilot:
        panel = app.query_one("#panel")
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert panel.display
        await pilot.resize_terminal(140, 30)
        await pilot.pause()
        assert panel.display
        await pilot.press("ctrl+t")
        await pilot.pause()
        assert not panel.display
        await pilot.resize_terminal(100, 30)
        await pilot.pause()
        assert not panel.display
        await pilot.press("ctrl+t")
        await pilot.resize_terminal(140, 30)
        await pilot.pause()
        assert panel.display


@pytest.mark.unit
async def test_the_column_is_wider_on_a_very_wide_screen(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(200, 40)) as pilot:
        app.view._message("ant", "reviewer", "task", "check a.py " * 20)
        await pilot.pause()
        await pilot.pause()
        assert app.query_one("#panel").region.width == 50
        lines, room = _rows(app)
        assert len(lines[1]) == room > 40


@pytest.mark.unit
async def test_a_long_plan_leaves_room_for_the_messages(tmp_path):
    from ant_ai.cli.render import PLAN_TOOL
    from ant_ai.core.message import ToolCall, ToolFunction

    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(120, 24)) as pilot:
        todos = [
            {"content": f"step {n}", "status": "completed" if n < 12 else "pending"}
            for n in range(30)
        ]
        args = json.dumps({"todos": todos})
        app.view._plan(
            ToolCall(id="1", function=ToolFunction(name=PLAN_TOOL, arguments=args))
        )
        app.view._message("ant", "reviewer", "task", "check a.py")
        await pilot.pause()
        plan = str(app.query_one("#side-plan", Static).render())
        # The current item is in sight, with what is before and after it counted.
        assert "step 12" in plan and "11 done before" in plan and "more" in plan
        panel = app.query_one("#panel").region
        assert app.query_one("#messages").region.bottom <= panel.bottom


@pytest.mark.unit
async def test_no_state_word_moves_the_context(tmp_path):
    app = AntApp(["sleep", "5"], tmp_path, label="scripted", context_size=200_000)
    async with app.run_test(size=(140, 30)) as pilot:
        await pilot.pause()
        places = []
        for state in ("connecting", "thinking", "running", "writing", "asking you"):
            app.view.doing(state)
            status = str(app.query_one("#status", Static).render())
            places.append(status.index("context"))
        assert len(set(places)) == 1, places


@pytest.mark.unit
async def test_wide_characters_stay_inside_the_message_rows(tmp_path):
    from rich.cells import cell_len

    app = AntApp(["sleep", "5"], tmp_path, label="scripted")
    async with app.run_test(size=(140, 30)) as pilot:
        app.view._message("サブ", "ant", "answer", "日本語のテキスト" * 10)
        await pilot.pause()
        await pilot.pause()
        lines, room = _rows(app)
        assert all(cell_len(line) <= room for line in lines), lines
        assert re.search(r"answer \d\d:\d\d$", lines[0])


@pytest.mark.unit
async def test_quitting_during_a_question_logs_no_error(tmp_path, caplog):
    app = AntApp(
        [sys.executable, str(AGENT), str(tmp_path)],
        tmp_path,
        label="scripted",
        max_steps=5,
    )
    async with app.run_test(size=(130, 36)) as pilot:
        await _ready(app)
        await pilot.click("#ask")
        await pilot.press(*"create a.txt", "enter")
        for _ in range(100):
            if isinstance(app.screen, PermissionScreen):
                break
            await pilot.pause(0.05)
        assert isinstance(app.screen, PermissionScreen)
        await pilot.press("ctrl+q")
        await pilot.pause(0.5)
    assert "Traceback" not in caplog.text
