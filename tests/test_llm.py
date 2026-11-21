from types import SimpleNamespace

import pytest

import llm


class FakeOpenAI:
    def __init__(self):
        self.calls = []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text="openai 요약")


def test_should_call_openai_summary_model_and_return_text():
    client = FakeOpenAI()

    text = llm.generate("openai", "summary", "시스템", "리포트", client=client)

    assert text == "openai 요약"
    assert client.calls[0]["model"] == "gpt-6-luna"
    assert client.calls[0]["instructions"] == "시스템"
    assert client.calls[0]["input"] == "리포트"


def test_should_use_larger_openai_model_for_insight():
    client = FakeOpenAI()

    llm.generate("openai", "insight", "시스템", "요약들", client=client)

    assert client.calls[0]["model"] == "gpt-6.2-sol"


def test_should_reject_unknown_provider():
    with pytest.raises(ValueError):
        llm.generate("gemini", "summary", "s", "u", client=object())


class FakeAnthropic:
    def __init__(self, stop_reason="end_turn"):
        self.calls = []
        self.stop_reason = stop_reason
        self.messages = SimpleNamespace(create=self._create("messages"))

    def _create(self, endpoint):
        def create(**kwargs):
            self.calls.append((endpoint, kwargs))
            content = [
                SimpleNamespace(type="thinking", thinking=""),
                SimpleNamespace(type="text", text="claude "),
                SimpleNamespace(type="text", text="요약"),
            ]
            return SimpleNamespace(content=content, stop_reason=self.stop_reason, stop_details=None)

        return create


def test_should_call_claude_summary_model_with_system_prompt():
    client = FakeAnthropic()

    text = llm.generate("claude", "summary", "시스템", "리포트", client=client)

    assert text == "claude 요약"
    endpoint, kwargs = client.calls[0]
    assert endpoint == "messages"
    assert kwargs["model"] == "claude-haiku-5-5"
    assert kwargs["system"] == "시스템"
    assert kwargs["messages"] == [{"role": "user", "content": "리포트"}]


def test_should_use_haiku_for_claude_insight_too():
    client = FakeAnthropic()

    llm.generate("claude", "insight", "시스템", "요약들", client=client)

    endpoint, kwargs = client.calls[0]
    assert endpoint == "messages"
    assert kwargs["model"] == "claude-haiku-5-5"


def test_should_raise_when_claude_refuses():
    with pytest.raises(RuntimeError):
        llm.generate("claude", "summary", "s", "u", client=FakeAnthropic(stop_reason="refusal"))


def test_should_read_api_key_env_name_per_provider():
    assert llm.API_KEY_ENV == {"openai": "TAPLY_API_KEY", "claude": "TAPLY_API_KEY"}


def test_should_send_openai_calls_through_gateway():
    client = llm.default_client("openai", api_key="test-key")

    assert str(client.base_url).rstrip("/") == "https://api.songseungwoon.com/v1"
    assert client.api_key == "test-key"


def test_should_send_claude_calls_through_gateway():
    client = llm.default_client("claude", api_key="test-key")

    assert str(client.base_url).rstrip("/") == "https://api.songseungwoon.com"
    assert client.api_key == "test-key"
