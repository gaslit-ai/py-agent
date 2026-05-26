"""Tests for adapters.llm — protocol shape and instructor-backed clients.

Real LLM calls aren't exercised here; we mock the underlying client surface and
verify the adapter passes through the right parameters and wraps errors.
A live Ollama integration test is marked and skipped unless OLLAMA_LIVE=1.
"""
from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
import respx
from pydantic import BaseModel, Field

from py_agent_lib.adapters.llm import StructuredLlm, from_ollama
from py_agent_lib.adapters.llm.base import LlmError
from py_agent_lib.adapters.llm.instructor_impl import (
    _AnthropicStyleLlm,
    _OpenAiStyleLlm,
    from_anthropic,
    from_litellm,
    from_openai,
)
from py_agent_lib.adapters.llm.ollama_native import _OllamaNativeLlm


class _Person(BaseModel):
    name: str
    age: int = Field(ge=0, le=150)


async def test_openai_style_extract_passes_messages_and_response_model() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_Person(name="Alice", age=32)
    )

    llm = _OpenAiStyleLlm("gpt-4o-mini", client=mock_client)
    result = await llm.extract(
        response_model=_Person,
        system="You are an extractor.",
        user="Alice, 32.",
    )

    assert isinstance(result, _Person)
    assert result.name == "Alice"

    call_kwargs = mock_client.chat.completions.create.await_args.kwargs
    assert call_kwargs["model"] == "gpt-4o-mini"
    assert call_kwargs["response_model"] is _Person
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "You are an extractor."},
        {"role": "user", "content": "Alice, 32."},
    ]
    assert call_kwargs["max_retries"] == 2


async def test_openai_style_extract_merges_defaults_and_kwargs() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        return_value=_Person(name="x", age=1)
    )

    llm = _OpenAiStyleLlm(
        "gpt-4o-mini",
        client=mock_client,
        default_kwargs={"temperature": 0.1, "top_p": 0.9},
    )
    await llm.extract(
        response_model=_Person,
        system="s",
        user="u",
        temperature=0.0,  # override default
    )

    kw = mock_client.chat.completions.create.await_args.kwargs
    assert kw["temperature"] == 0.0  # caller wins
    assert kw["top_p"] == 0.9  # default preserved


async def test_openai_style_wraps_underlying_errors() -> None:
    mock_client = MagicMock()
    mock_client.chat.completions.create = AsyncMock(
        side_effect=ConnectionError("network down")
    )
    llm = _OpenAiStyleLlm("gpt-4o-mini", client=mock_client)

    with pytest.raises(LlmError, match="network down") as ei:
        await llm.extract(response_model=_Person, system="s", user="u")
    assert isinstance(ei.value.__cause__, ConnectionError)


async def test_anthropic_style_uses_messages_create_with_top_level_system() -> None:
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_Person(name="x", age=1))

    llm = _AnthropicStyleLlm("claude-sonnet-4", client=mock_client)
    await llm.extract(response_model=_Person, system="sys", user="usr")

    kw = mock_client.messages.create.await_args.kwargs
    assert kw["system"] == "sys"
    assert kw["messages"] == [{"role": "user", "content": "usr"}]
    assert kw["max_tokens"] == 4096  # default


async def test_anthropic_style_max_tokens_override() -> None:
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock(return_value=_Person(name="x", age=1))

    llm = _AnthropicStyleLlm(
        "claude-sonnet-4",
        client=mock_client,
        default_kwargs={"max_tokens": 1024},
    )
    await llm.extract(response_model=_Person, system="s", user="u", max_tokens=512)

    kw = mock_client.messages.create.await_args.kwargs
    assert kw["max_tokens"] == 512


def test_factory_from_openai_returns_openai_style() -> None:
    llm = from_openai("gpt-4o-mini", api_key="sk-test")
    assert isinstance(llm, _OpenAiStyleLlm)
    assert llm.model == "gpt-4o-mini"


def test_factory_from_anthropic_returns_anthropic_style() -> None:
    llm = from_anthropic("claude-sonnet-4", api_key="sk-test")
    assert isinstance(llm, _AnthropicStyleLlm)
    assert llm.model == "claude-sonnet-4"


def test_factory_from_litellm_returns_openai_style() -> None:
    llm = from_litellm("ollama/gemma4")
    assert isinstance(llm, _OpenAiStyleLlm)
    assert llm.model == "ollama/gemma4"


def test_structured_llm_protocol_satisfied_by_implementations() -> None:
    """Both client classes should structurally satisfy the StructuredLlm protocol."""
    openai_llm = _OpenAiStyleLlm("m", MagicMock())
    anthropic_llm = _AnthropicStyleLlm("m", MagicMock())
    assert isinstance(openai_llm, StructuredLlm)
    assert isinstance(anthropic_llm, StructuredLlm)


# ----- Ollama native client -----


def test_factory_from_ollama_returns_native_client() -> None:
    llm = from_ollama("gemma4", host="http://192.168.0.10:11434")
    assert isinstance(llm, _OllamaNativeLlm)
    assert llm.model == "gemma4"
    assert llm._host == "http://192.168.0.10:11434"


async def test_ollama_native_basic_extract() -> None:
    with respx.mock() as mock:
        mock.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200, json={"message": {"content": '{"name":"Alice","age":32}'}}
            )
        )
        llm = from_ollama("gemma4")
        result = await llm.extract(response_model=_Person, system="s", user="u")
    assert result.name == "Alice"
    assert result.age == 32


async def test_ollama_native_sends_format_schema_in_body() -> None:
    with respx.mock() as mock:
        route = mock.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200, json={"message": {"content": '{"name":"x","age":1}'}}
            )
        )
        llm = from_ollama("gemma4")
        await llm.extract(response_model=_Person, system="sys", user="usr")

    body = json.loads(route.calls[0].request.content)
    assert body["model"] == "gemma4"
    assert body["stream"] is False
    assert body["format"]["type"] == "object"
    assert "name" in body["format"]["properties"]
    assert "age" in body["format"]["properties"]
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]


async def test_ollama_native_think_goes_to_top_level() -> None:
    with respx.mock() as mock:
        route = mock.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200, json={"message": {"content": '{"name":"x","age":1}'}}
            )
        )
        llm = from_ollama("gemma4", think=False)
        await llm.extract(response_model=_Person, system="s", user="u")

    body = json.loads(route.calls[0].request.content)
    assert body["think"] is False
    assert "think" not in body.get("options", {})


async def test_ollama_native_temperature_goes_to_options() -> None:
    with respx.mock() as mock:
        route = mock.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(
                200, json={"message": {"content": '{"name":"x","age":1}'}}
            )
        )
        llm = from_ollama("gemma4", temperature=0.0, top_p=0.9)
        await llm.extract(response_model=_Person, system="s", user="u")

    body = json.loads(route.calls[0].request.content)
    assert body["options"]["temperature"] == 0.0
    assert body["options"]["top_p"] == 0.9
    assert "temperature" not in body or body.get("temperature") is None


async def test_ollama_native_retries_on_validation_failure_and_succeeds() -> None:
    with respx.mock() as mock:
        route = mock.post("http://localhost:11434/api/chat").mock(
            side_effect=[
                # First response: missing required `age` field.
                httpx.Response(200, json={"message": {"content": '{"name":"Alice"}'}}),
                # Second response: valid.
                httpx.Response(
                    200, json={"message": {"content": '{"name":"Alice","age":32}'}}
                ),
            ]
        )
        llm = from_ollama("gemma4")
        result = await llm.extract(
            response_model=_Person, system="s", user="u", max_retries=1
        )
    assert result.name == "Alice"
    assert result.age == 32
    assert route.call_count == 2


async def test_ollama_native_appends_error_feedback_on_retry() -> None:
    """On retry, the previous bad output + an error-feedback turn should be in the prompt."""
    with respx.mock() as mock:
        route = mock.post("http://localhost:11434/api/chat").mock(
            side_effect=[
                httpx.Response(200, json={"message": {"content": '{"name":"x"}'}}),
                httpx.Response(
                    200, json={"message": {"content": '{"name":"x","age":1}'}}
                ),
            ]
        )
        llm = from_ollama("gemma4")
        await llm.extract(response_model=_Person, system="s", user="u", max_retries=1)

    second_body = json.loads(route.calls[1].request.content)
    msgs = second_body["messages"]
    # original system + user, plus the bad assistant turn and the corrective user turn
    assert len(msgs) == 4
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == '{"name":"x"}'
    assert msgs[3]["role"] == "user"
    assert "failed JSON-schema validation" in msgs[3]["content"]


async def test_ollama_native_raises_after_max_retries() -> None:
    with respx.mock() as mock:
        mock.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(200, json={"message": {"content": "{}"}})
        )
        llm = from_ollama("gemma4")
        with pytest.raises(LlmError, match="failed validation"):
            await llm.extract(
                response_model=_Person, system="s", user="u", max_retries=1
            )


async def test_ollama_native_wraps_http_error() -> None:
    with respx.mock() as mock:
        mock.post("http://localhost:11434/api/chat").mock(
            return_value=httpx.Response(500, text="server error")
        )
        llm = from_ollama("gemma4")
        with pytest.raises(LlmError, match="HTTP"):
            await llm.extract(response_model=_Person, system="s", user="u")


async def test_ollama_native_custom_host() -> None:
    with respx.mock() as mock:
        route = mock.post("http://192.168.1.5:11434/api/chat").mock(
            return_value=httpx.Response(
                200, json={"message": {"content": '{"name":"x","age":1}'}}
            )
        )
        llm = from_ollama("gemma4", host="http://192.168.1.5:11434")
        await llm.extract(response_model=_Person, system="s", user="u")
    assert route.called


# ----- live Ollama integration (only when OLLAMA_LIVE=1) -----


@pytest.mark.skipif(
    os.environ.get("OLLAMA_LIVE") != "1",
    reason="set OLLAMA_LIVE=1 to run live Ollama integration tests",
)
async def test_live_ollama_extract_native() -> None:
    """Live test: native /api/chat with gemma4 + think=False for sub-second response."""
    llm = from_ollama("gemma4", think=False)
    person = await llm.extract(
        response_model=_Person,
        system="Extract person info as JSON.",
        user="Alice is 32 years old.",
    )
    assert person.name.lower().startswith("alice")
    assert 0 <= person.age <= 150
