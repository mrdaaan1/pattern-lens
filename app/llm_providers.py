"""
Единая точка вызова LLM для анализа транскрибаций — с переключателем провайдера.

Поддерживаются два провайдера:
• gigachat    — через langchain-gigachat (как в smartmove-agent-py)
• openrouter  — прямой REST-вызов OpenRouter (ключ переиспользован из smartmove-copilot/.env.local)

Список моделей — фиксированный, чтобы в UI был понятный выпадающий список, а не свободный ввод
строки (легко ошибиться в id модели и получить непонятную 400-ошибку от апстрима).
"""

import asyncio

import httpx
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_gigachat.chat_models import GigaChat

from app.config import get_settings

# Апстрим-провайдеры OpenRouter (DeepInfra, Novita и т.д.) иногда обрывают соединение
# без ответа при генерации длинного ответа (httpx.RemoteProtocolError: "peer closed
# connection without sending complete message body") — воспроизведено живьём на длинной
# транскрибации (~10 страниц). Это не ошибка запроса, а временный сбой конкретного
# провайдера, поэтому повторяем, как в smartmove-copilot/src/lib/openrouter.ts.
# ВАЖНО: таймаут одной попытки НЕ должен вплотную приближаться к общему дедлайну
# (_TOTAL_DEADLINE_S) — иначе первая же зависшая попытка съедает весь бюджет и retry
# просто не успевает случиться. Наблюдалось живьём: с попыткой в 120с и 3 retry общее
# время молчания для пользователя доходило до ~7 минут без какой-либо обратной связи.
_OPENROUTER_ATTEMPT_TIMEOUT_S = 45.0
_OPENROUTER_MAX_RETRIES = 2
_OPENROUTER_RETRY_DELAY_S = 1.5

# Дефолтный таймаут langchain-gigachat (обычно ~30с на HTTP-запрос) не хватает на длинную
# транскрибацию (~30к символов) — воспроизведено живьём: httpx.ReadTimeout на GigaChat-2-Max.
# Поднимаем таймаут, но держим его заметно меньше общего дедлайна — по той же причине.
_GIGACHAT_ATTEMPT_TIMEOUT_S = 90.0
_GIGACHAT_MAX_RETRIES = 1
_GIGACHAT_RETRY_DELAY_S = 1.5

# Жёсткий потолок на весь вызов (включая все retry) — после него пользователь получает
# понятную ошибку вместо бесконечного "Анализирую...". На длинные транскрибации (~30к
# символов) GigaChat-2-Max реально может отвечать ~70-90с за один успешный запрос — 150с
# оставляет запас на одну зависшую попытку + один успешный повтор.
_TOTAL_DEADLINE_S = 150.0

GIGACHAT_MODELS = ["GigaChat-2", "GigaChat-2-Pro", "GigaChat-2-Max"]

OPENROUTER_MODELS = [
    "deepseek/deepseek-chat",
    "google/gemini-2.5-flash-lite",
    "google/gemini-2.5-flash",
    "anthropic/claude-sonnet-4.5",
    "openai/gpt-4.1-mini",
    "qwen/qwen3-235b-a22b",
]

PROVIDERS = {
    "gigachat": GIGACHAT_MODELS,
    "openrouter": OPENROUTER_MODELS,
}

DEFAULT_PROVIDER = "gigachat"
DEFAULT_MODEL = GIGACHAT_MODELS[0]


class LLMConfigError(Exception):
    pass


async def _call_gigachat(system_prompt: str, user_content: str, model: str) -> str:
    settings = get_settings()
    if not settings.gigachat_credentials:
        raise LLMConfigError("GIGACHAT_CREDENTIALS не заданы в .env")

    llm = GigaChat(
        credentials=settings.gigachat_credentials,
        scope=settings.gigachat_scope,
        model=model,
        verify_ssl_certs=settings.gigachat_verify_ssl_certs,
        base_url=settings.gigachat_base_url,
        timeout=_GIGACHAT_ATTEMPT_TIMEOUT_S,
    )

    last_error: Exception | None = None
    for attempt in range(_GIGACHAT_MAX_RETRIES + 1):
        try:
            response = await llm.ainvoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_content)]
            )
            return str(response.content)
        except httpx.HTTPError as e:
            last_error = LLMConfigError(f"GigaChat: сбой соединения с апстримом ({e})")
            if attempt == _GIGACHAT_MAX_RETRIES:
                raise last_error
            await asyncio.sleep(_GIGACHAT_RETRY_DELAY_S * (attempt + 1))

    raise last_error or LLMConfigError("GigaChat: не удалось получить ответ")


async def _call_openrouter(system_prompt: str, user_content: str, model: str) -> str:
    settings = get_settings()
    if not settings.openrouter_api_key:
        raise LLMConfigError("OPENROUTER_API_KEY не задан в .env")

    last_error: Exception | None = None

    for attempt in range(_OPENROUTER_MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=_OPENROUTER_ATTEMPT_TIMEOUT_S) as client:
                res = await client.post(
                    f"{settings.openrouter_base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {settings.openrouter_api_key}",
                        "Content-Type": "application/json",
                        "HTTP-Referer": "http://localhost",
                        "X-Title": "Pattern Lens",
                    },
                    json={
                        "model": model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_content},
                        ],
                        "temperature": 0.2,
                        "max_tokens": 16000,
                    },
                )
                if res.status_code == 200:
                    data = res.json()
                    return data["choices"][0]["message"]["content"] or ""

                is_retryable = res.status_code == 429 or res.status_code >= 500
                last_error = LLMConfigError(f"OpenRouter error {res.status_code}: {res.text}")
                if not is_retryable or attempt == _OPENROUTER_MAX_RETRIES:
                    raise last_error
        except httpx.HTTPError as e:
            # Обрыв соединения / таймаут апстрим-провайдера — не ошибка самого запроса.
            last_error = LLMConfigError(f"OpenRouter: сбой соединения с апстрим-провайдером ({e})")
            if attempt == _OPENROUTER_MAX_RETRIES:
                raise last_error

        await asyncio.sleep(_OPENROUTER_RETRY_DELAY_S * (attempt + 1))

    raise last_error or LLMConfigError("OpenRouter: не удалось получить ответ")


async def call_llm(system_prompt: str, user_content: str, provider: str, model: str) -> str:
    if provider not in PROVIDERS:
        raise LLMConfigError(f"Неизвестный провайдер: {provider}")
    if model not in PROVIDERS[provider]:
        raise LLMConfigError(f"Модель {model} недоступна для провайдера {provider}")

    if provider == "gigachat":
        coro = _call_gigachat(system_prompt, user_content, model)
    else:
        coro = _call_openrouter(system_prompt, user_content, model)

    try:
        return await asyncio.wait_for(coro, timeout=_TOTAL_DEADLINE_S)
    except asyncio.TimeoutError:
        raise LLMConfigError(
            f"{provider}: превышен лимит ожидания ответа ({int(_TOTAL_DEADLINE_S)}с). "
            "Попробуйте более быструю модель или сократите текст транскрибации."
        )
