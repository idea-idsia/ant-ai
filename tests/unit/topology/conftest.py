from __future__ import annotations

import pytest
from fakes import StubLLM


@pytest.fixture
def stub_llm() -> StubLLM:
    return StubLLM()
