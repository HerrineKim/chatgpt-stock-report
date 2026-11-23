import os
import time

import anthropic
import openai
from openai import OpenAI

# task별 모델: summary는 리포트마다(하루 100건 안팎) 부르고, insight는 하루 한 번 부른다.
# 모델명은 TAPLY 게이트웨이 등급표를 따른다(lite: gpt-6-luna·claude-haiku-5-5, pro: gpt-6.2-sol).
MODELS = {
    "openai": {"summary": "gpt-6-luna", "insight": "gpt-6.2-sol"},
    "claude": {"summary": "claude-haiku-5-5", "insight": "claude-haiku-5-5"},
}
# 두 제공사 모두 TAPLY 게이트웨이를 거치므로 키 하나로 부른다.
API_KEY_ENV = {"openai": "TAPLY_API_KEY", "claude": "TAPLY_API_KEY"}
GATEWAY_BASE_URL = {
    "openai": "https://api.songseungwoon.com/v1",
    "claude": "https://api.songseungwoon.com",
}


def default_client(provider, api_key=None):
    """provider SDK 클라이언트를 게이트웨이 주소로 만든다."""
    if provider == "claude":
        return anthropic.Anthropic(api_key=api_key, base_url=GATEWAY_BASE_URL["claude"])
    return OpenAI(api_key=api_key, base_url=GATEWAY_BASE_URL["openai"])


# 게이트웨이가 막혔을 때(아래 is_gateway_blocked) 제공사 API를 직접 부르는 데 쓰는 키와 모델명.
# GitHub Actions 러너 IP는 게이트웨이 앞 Cloudflare의 봇 챌린지에 걸릴 수 있다.
DIRECT_API_KEY_ENV = {"openai": "OPEN_API_KEY", "claude": "ANTHROPIC_API_KEY"}
DIRECT_MODELS = {
    "openai": {"summary": "gpt-5-mini", "insight": "gpt-5"},
    "claude": MODELS["claude"],
}
_gateway_blocked = False  # 한 번 막히면 같은 실행의 남은 호출은 게이트웨이를 건너뛴다


def direct_client(provider, api_key):
    """provider SDK 클라이언트를 제공사 기본 주소로 만든다."""
    if provider == "claude":
        return anthropic.Anthropic(api_key=api_key)
    return OpenAI(api_key=api_key)


def is_gateway_blocked(error):
    """게이트웨이 서버에 닿기 전에 막힌 오류(Cloudflare 챌린지, 연결 실패)인지 본다."""
    if isinstance(error, (anthropic.APIConnectionError, openai.APIConnectionError)):
        return True
    headers = getattr(getattr(error, "response", None), "headers", None) or {}
    return headers.get("cf-mitigated") == "challenge"


RETRIES = 3  # OpenAI 호출 실패 시 재시도 횟수


def generate(provider, task, system_prompt, user_prompt, api_key=None, client=None):
    """provider의 task 모델로 system/user 프롬프트를 보내고 응답 텍스트를 돌려준다."""
    if provider not in MODELS:
        raise ValueError(f"Unknown provider: {provider} (choose one of {', '.join(MODELS)})")
    if client is not None:
        return _call(provider, MODELS[provider][task], system_prompt, user_prompt, client)

    global _gateway_blocked
    if not _gateway_blocked:
        try:
            return _call(provider, MODELS[provider][task], system_prompt, user_prompt, default_client(provider, api_key))
        except Exception as e:
            if not is_gateway_blocked(e) or not os.environ.get(DIRECT_API_KEY_ENV[provider]):
                raise
            _gateway_blocked = True
            print(f"Gateway blocked ({type(e).__name__}). Falling back to the direct {provider} API for this run.")
    direct = direct_client(provider, os.environ[DIRECT_API_KEY_ENV[provider]])
    return _call(provider, DIRECT_MODELS[provider][task], system_prompt, user_prompt, direct)


def _call(provider, model, system_prompt, user_prompt, client):
    if provider == "claude":
        return _claude(model, system_prompt, user_prompt, client)
    return _openai(model, system_prompt, user_prompt, client)


def _claude(model, system_prompt, user_prompt, client):
    # 재시도(429, 5xx, 연결 오류)는 SDK가 기본으로 처리한다.
    response = client.messages.create(
        model=model,
        max_tokens=16000,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    if response.stop_reason == "refusal":
        category = getattr(response.stop_details, "category", None)
        raise RuntimeError(f"{model} refused the request (category: {category})")
    return "".join(block.text for block in response.content if block.type == "text")


def _openai(model, system_prompt, user_prompt, client):
    # 게이트웨이 안내가 Responses API 기준이라 Chat Completions 대신 이것을 쓴다.
    backoff_time = 10
    for attempt in range(RETRIES + 1):
        try:
            response = client.responses.create(model=model, instructions=system_prompt, input=user_prompt)
            return response.output_text
        except Exception as e:
            if attempt == RETRIES or is_gateway_blocked(e):
                raise
            print(f"{type(e).__name__}: {e}. Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 1.5
