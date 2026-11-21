import time

import anthropic
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


RETRIES = 3  # OpenAI 호출 실패 시 재시도 횟수


def generate(provider, task, system_prompt, user_prompt, api_key=None, client=None):
    """provider의 task 모델로 system/user 프롬프트를 보내고 응답 텍스트를 돌려준다."""
    if provider not in MODELS:
        raise ValueError(f"Unknown provider: {provider} (choose one of {', '.join(MODELS)})")
    model = MODELS[provider][task]
    client = client or default_client(provider, api_key)
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
            if attempt == RETRIES:
                raise
            print(f"{type(e).__name__}: {e}. Retrying in {backoff_time} seconds...")
            time.sleep(backoff_time)
            backoff_time *= 1.5
