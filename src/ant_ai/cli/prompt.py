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

    Enter sends; Ctrl-C or Ctrl-D on an empty box returns None. The box is
    erased afterwards, so the caller prints the message in the transcript.
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

    @keys.add("c-c")
    @keys.add("c-d")
    def _quit(event: KeyPressEvent) -> None:
        if area.text and event.key_sequence[0].key == "c-c":
            area.text = ""
            return
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
    """Ask a one-line question (e.g. a confirmation) and return the answer."""
    try:
        return await PromptSession().prompt_async(
            [("#777777", "  ⎿ "), (f"{ACCENT} bold", f"{question} ")], style=STYLE
        )
    except KeyboardInterrupt, EOFError:
        return ""
