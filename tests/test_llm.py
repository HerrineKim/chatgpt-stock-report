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


class ChallengedClient:
    """게이트웨이 앞 Cloudflare가 챌린지 페이지를 돌려주는 상황을 흉내 낸다."""

    def __init__(self, log):
        def create(**kwargs):
            log.append("gateway")
            error = RuntimeError("<title>Just a moment...</title>")
            error.response = SimpleNamespace(headers={"cf-mitigated": "challenge"})
            raise error

        self.messages = SimpleNamespace(create=create)


@pytest.fixture
def blocked_gateway(monkeypatch):
    log = []
    monkeypatch.setattr(llm, "_gateway_blocked", False)
    monkeypatch.setattr(llm, "default_client", lambda provider, api_key=None: ChallengedClient(log))
    return log


def test_should_fall_back_to_direct_api_when_gateway_returns_cloudflare_challenge(monkeypatch, blocked_gateway):
    direct = FakeAnthropic()
    direct_keys = []
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-key")
    monkeypatch.setattr(llm, "direct_client", lambda provider, api_key: direct_keys.append(api_key) or direct)

    text = llm.generate("claude", "summary", "시스템", "리포트", api_key="taply-key")

    assert text == "claude 요약"
    assert direct_keys == ["direct-key"]
    assert direct.calls[0][1]["model"] == "claude-haiku-5-5"


def test_should_skip_gateway_after_it_was_blocked_once(monkeypatch, blocked_gateway):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-key")
    monkeypatch.setattr(llm, "direct_client", lambda provider, api_key: FakeAnthropic())

    llm.generate("claude", "summary", "s", "u", api_key="taply-key")
    llm.generate("claude", "summary", "s", "u", api_key="taply-key")

    assert blocked_gateway == ["gateway"]


def test_should_raise_gateway_error_when_no_direct_key_is_set(monkeypatch, blocked_gateway):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="Just a moment"):
        llm.generate("claude", "summary", "s", "u", api_key="taply-key")


def test_should_not_fall_back_on_ordinary_api_errors(monkeypatch):
    class FailingClient:
        messages = SimpleNamespace(create=lambda **kwargs: (_ for _ in ()).throw(RuntimeError("invalid model")))

    monkeypatch.setattr(llm, "_gateway_blocked", False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "direct-key")
    monkeypatch.setattr(llm, "default_client", lambda provider, api_key=None: FailingClient())

    with pytest.raises(RuntimeError, match="invalid model"):
        llm.generate("claude", "summary", "s", "u", api_key="taply-key")


def test_should_build_direct_clients_without_gateway_base_url():
    claude = llm.direct_client("claude", "direct-key")
    openai_client = llm.direct_client("openai", "direct-key")

    assert "songseungwoon" not in str(claude.base_url)
    assert "songseungwoon" not in str(openai_client.base_url)
    assert claude.api_key == "direct-key"


def test_should_use_provider_model_names_when_calling_openai_directly():
    assert llm.DIRECT_MODELS["openai"] == {"summary": "gpt-5-mini", "insight": "gpt-5"}
