from __future__ import annotations

import pytest

from ant_ai.acp.commands import ACPCommand, parse_slash_command


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/skill install greet", ("skill", "install greet")),
        ("/compact", ("compact", "")),
        ("  /compact  ", ("compact", "")),
        ("/run\nmulti line body", ("run", "multi line body")),
        ("/plan   ship it  ", ("plan", "ship it")),
        ("plain text, no command", (None, "")),
        ("/etc/hosts", (None, "")),  # bare path, not a command
        ("/", (None, "")),
        ("", (None, "")),
        ("please run /compact later", (None, "")),  # command must be leading
    ],
)
def test_parse_slash_command(text: str, expected: tuple[str | None, str]) -> None:
    assert parse_slash_command(text) == expected


def test_code_command_requires_handler() -> None:
    with pytest.raises(ValueError, match="requires a handler"):
        ACPCommand(name="x", description="d", kind="code")


def test_code_command_with_handler_ok() -> None:
    async def _h(_ctx):  # noqa: ANN001
        return None

    cmd = ACPCommand(name="x", description="d", kind="code", handler=_h)
    assert cmd.handler is _h


def test_expand_with_template() -> None:
    cmd = ACPCommand(
        name="plan", description="d", kind="prompt", template="Plan for:\n{args}"
    )
    assert cmd.expand("ship it") == "Plan for:\nship it"


def test_expand_without_template_is_identity() -> None:
    assert (
        ACPCommand(name="p", description="d").expand("hello {world}") == "hello {world}"
    )


def test_expand_leaves_braces_in_args_untouched() -> None:
    cmd = ACPCommand(name="p", description="d", kind="prompt", template="<<{args}>>")
    assert cmd.expand('{"k": 1}') == '<<{"k": 1}>>'


def test_to_available_command_carries_hint() -> None:
    ac = ACPCommand(
        name="skill", description="manage skills", input_hint="install <name>"
    ).to_available_command()
    assert ac.name == "skill"
    assert ac.description == "manage skills"
    assert ac.input is not None
    assert ac.input.root.hint == "install <name>"


def test_to_available_command_without_hint() -> None:
    ac = ACPCommand(name="compact", description="d").to_available_command()
    assert ac.input is None
