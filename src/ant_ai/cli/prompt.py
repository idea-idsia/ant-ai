from __future__ import annotations

from prompt_toolkit.application import Application
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings, KeyPressEvent
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.layout.processors import AfterInput, ConditionalProcessor
from prompt_toolkit.shortcuts import PromptSession
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Frame, TextArea

from ant_ai.cli.render import ACCENT

STYLE = Style.from_dict(
    {
        "frame.border": ACCENT,
        "placeholder": "#777777",
        "hint": "#777777",
    }
)


async def read_message(hint: str) -> str | None:
    """Show the input box at the bottom and return what the user typed.

    Enter sends; Ctrl-C clears the box, or on an empty box quits like Ctrl-D.
    The box is erased afterwards, so the caller prints the message in the
    transcript.

    Args:
        hint: The line under the box (model, folder, `/help`).

    Returns:
        The text typed, or None when the user quits.
    """
    area = TextArea(
        prompt="› ",
        multiline=False,
        wrap_lines=True,
        input_processors=[
            ConditionalProcessor(
                AfterInput("ask ant to do something", style="class:placeholder"),
                filter=Condition(lambda: not area.text),
            )
        ],
    )
    keys = KeyBindings()
    empty = Condition(lambda: not area.text)

    @keys.add("c-c")
    def _cancel(event: KeyPressEvent) -> None:
        if area.text:
            area.text = ""
            return
        event.app.exit(result=None)

    # With text in the box, Ctrl-D keeps its usual meaning: delete a character.
    @keys.add("c-d", filter=empty)
    def _quit(event: KeyPressEvent) -> None:
        event.app.exit(result=None)

    def _send(buffer) -> bool:
        app.exit(result=buffer.text)
        return True

    area.accept_handler = _send
    app: Application[str | None] = Application(
        layout=Layout(
            HSplit(
                [
                    Frame(area),
                    Window(
                        FormattedTextControl([("class:hint", f" {hint}")]), height=1
                    ),
                ]
            ),
            focused_element=area,
        ),
        key_bindings=keys,
        style=STYLE,
        erase_when_done=True,
    )
    return await app.run_async()


async def read_answer(question: str) -> str:
    """Ask a one-line question, e.g. a confirmation.

    Args:
        question: Shown before the cursor.

    Returns:
        The answer, or "" on Ctrl-C or Ctrl-D.
    """
    try:
        return await PromptSession().prompt_async(
            [("#777777", "  ⎿ "), (f"{ACCENT} bold", f"{question} ")], style=STYLE
        )
    except KeyboardInterrupt, EOFError:
        return ""
