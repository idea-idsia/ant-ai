from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from textual.widgets import Collapsible, DataTable, Markdown, Static, Tree

from ant_ai.cli.tui import ANT, AntApp, PermissionScreen

AGENT = Path(__file__).with_name("scripted_acp_agent.py")


async def _run_turn(app: AntApp, pilot, answer: str) -> None:
    await app._connected.wait()
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
        assert "Wrote a.txt." in app.query_one(Markdown).source
        table = app.query_one("#messages", DataTable)
        rows = [table.get_row_at(i) for i in range(table.row_count)]
        # Only messages between agents: the tool call stays in the conversation.
        assert [(r[1], r[2]) for r in rows] == [("user", "ant"), ("ant", "user")]
        context = str(app.query_one("#context", Static).render())
        assert "≈" in context and "estimate" in context
        agents = app.query_one("#agents", Tree)
        assert "waiting" in str(agents.root.children[0].label)
        assert str(call.title).endswith("s")
        assert "2/5" in str(app.query_one("#tools", Static).render())
        assert "write_file" in str(app.query_one("#tools", Static).render())
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
        assert "1 ✗1" in str(app.query_one("#tools", Static).render())
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
            assert str(app.query_one(".live", Static).render()) == "Hello"
            status = str(app.query_one("#status", Static).render())
            assert status.startswith("■ writing")
            # The ant walks on its own, at the right: the status text does not move it.
            walker = app.query_one("#walker", Static)
            assert any(str(walker.render()).startswith(ant) for ant in ANT)
            assert walker.region.right == app.query_one("#ask").region.right
            app.view.event(FinalAnswerEvent(content="Hello"))
            await pilot.pause()
            assert not app.query(".live")
        assert str(app.query_one("#status", Static).render()) == "■ ready"


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
        await app._connected.wait()
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
        await app._connected.wait()
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
            assert [str(w.render()) for w in app.query(".live")] == ["Now I read one."]


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
