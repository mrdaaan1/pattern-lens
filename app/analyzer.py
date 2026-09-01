import json
import logging
import re
from typing import Any

from pydantic import BaseModel

from app.llm_providers import DEFAULT_MODEL, DEFAULT_PROVIDER, call_llm
from app.stages import STAGES, Stage

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """Ты — аналитик супервизии, который проверяет транскрибации встреч "бодрое утро" \
(планёрка руководителя точки продаж малого бизнеса с клиентскими менеджерами) на соответствие \
фиксированной методологии оценки, разбитой на ЭТАПЫ. Каждый этап состоит из нескольких КРИТЕРИЕВ, \
и по каждому критерию нужно вынести вердикт Да/Нет с пояснением.

Тебе дают:
1. Пронумерованный список этапов методологии, каждый — с критериями (что именно считается выполненным) \
и правилом агрегации этапа.
2. Полный текст транскрибации.

Твоя задача:
1. Для КАЖДОГО критерия каждого этапа вынести вердикт: verdict = "yes" или "no", explanation — короткое \
пояснение на русском, ГДЕ ИМЕННО в диалоге (в каком смысловом блоке/эпизоде) это обнаружено (если "yes") \
или почему критерий не выполнен (если "no"; например "в тексте не упомянуто", "передачи слова не было"). \
Следуй ТОЧНО описанию критерия — включая явно расписанные исключения (например «если релиза не было и это \
озвучено — засчитываем») и явно расписанные случаи «считать выполненным при отсутствии данных».
2. Если критерий выполнен ("yes"), найди в тексте ОДНУ дословную цитату (quote), которая подтверждает вердикт \
— минимально достаточную, чтобы однозначно найти её через поиск подстроки в исходном тексте. Если критерий \
считается выполненным без опоры на конкретный фрагмент текста (например, по правилу «нет данных за прошлые \
дни — считаем выполненным»), оставь quote пустой строкой.
3. Напиши краткое итоговое заключение по всей транскрибации (3-6 предложений на русском): какие этапы \
выполнены полностью, какие — нет и почему, что стоит улучшить руководителю в проведении таких встреч. \
Опирайся только на то, что реально есть в тексте, не выдумывай.

Верни результат СТРОГО в формате JSON без каких-либо пояснений до или после, без markdown-обёртки \
(без ```json), в виде одного объекта:

{
  "criteria": [
    {
      "criterion_id": "<id критерия из списка, например news-1>",
      "verdict": "yes" | "no",
      "quote": "<дословная цитата из текста или пустая строка>",
      "explanation": "<короткое пояснение на русском>"
    }
  ],
  "summary": "<итоговое заключение по всей транскрибации>"
}

Правила:
• В "criteria" ОБЯЗАН быть ровно один объект на каждый criterion_id из присланного списка критериев — не пропускай ни одного.
• "quote" (если не пустая) ОБЯЗАНА быть точной подстрокой исходного текста транскрибации, без изменений (без исправления опечаток, без сокращений многоточием).
• Не выдумывай цитаты, которых нет в тексте."""


class CriterionVerdict(BaseModel):
    criterion_id: str
    criterion_text: str
    verdict: bool
    quote: str
    explanation: str
    start: int | None = None
    end: int | None = None


class StageResult(BaseModel):
    stage_id: str
    stage_title: str
    passed: bool
    criteria: list[CriterionVerdict]


class AnalysisResult(BaseModel):
    stages: list[StageResult]
    summary: str


def _build_stages_block(stages: list[Stage]) -> str:
    lines = []
    for stage in stages:
        lines.append(f"## Этап id={stage.id}: {stage.title}\nАгрегация: {stage.aggregation}")
        if stage.recommendation_rule:
            lines.append(f"Рекомендации: {stage.recommendation_rule}")
        for c in stage.criteria:
            lines.append(f"- Критерий id={c.id}: {c.text}\n  {c.detail}")
    return "\n\n".join(lines)


def _extract_json_object(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    raw = re.sub(r"^```(json)?", "", raw).strip()
    raw = re.sub(r"```$", "", raw).strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _locate_quote(transcript: str, quote: str) -> tuple[int, int] | None:
    if not quote:
        return None
    start = transcript.find(quote)
    if start == -1:
        start = transcript.lower().find(quote.lower())
        if start == -1:
            return None
    return start, start + len(quote)


async def analyze_transcript(
    transcript: str,
    provider: str = DEFAULT_PROVIDER,
    model: str = DEFAULT_MODEL,
) -> AnalysisResult:
    criteria_by_id = {c.id: (stage, c) for stage in STAGES for c in stage.criteria}

    stages_block = _build_stages_block(STAGES)
    user_content = f"Этапы и критерии методологии:\n\n{stages_block}\n\nТекст транскрибации:\n\n{transcript}"

    raw_response = await call_llm(_SYSTEM_PROMPT, user_content, provider, model)
    parsed = _extract_json_object(raw_response)
    raw_criteria = parsed.get("criteria", [])
    summary = str(parsed.get("summary", "")).strip()

    if not isinstance(raw_criteria, list):
        raw_criteria = []

    if not parsed:
        logger.warning(
            "LLM (%s/%s) вернул нераспознанный ответ (не JSON-объект). Начало ответа: %r",
            provider, model, raw_response[:300],
        )
        summary = summary or "Не удалось разобрать ответ модели. Попробуйте другую модель или повторите запрос."

    verdict_by_criterion: dict[str, CriterionVerdict] = {}
    for item in raw_criteria:
        try:
            criterion_id = str(item["criterion_id"])
            verdict = str(item.get("verdict", "no")).strip().lower() == "yes"
            quote = str(item.get("quote", "") or "")
            explanation = str(item.get("explanation", ""))
        except (KeyError, TypeError):
            continue

        found = criteria_by_id.get(criterion_id)
        if found is None:
            continue
        _stage, criterion = found

        located = _locate_quote(transcript, quote)
        start, end = located if located else (None, None)
        quote = transcript[start:end] if located else ""

        verdict_by_criterion[criterion_id] = CriterionVerdict(
            criterion_id=criterion_id,
            criterion_text=criterion.text,
            verdict=verdict,
            quote=quote,
            explanation=explanation,
            start=start,
            end=end,
        )

    stage_results: list[StageResult] = []
    for stage in STAGES:
        criteria_results = []
        for c in stage.criteria:
            v = verdict_by_criterion.get(c.id)
            if v is None:
                v = CriterionVerdict(
                    criterion_id=c.id,
                    criterion_text=c.text,
                    verdict=False,
                    quote="",
                    explanation="Модель не вернула вердикт по этому критерию.",
                )
            criteria_results.append(v)
        required_ids = {c.id for c in stage.criteria if not c.is_exception}
        required_verdicts = [v.verdict for v in criteria_results if v.criterion_id in required_ids]
        if stage.aggregation_mode == "any":
            passed = any(required_verdicts)
        else:
            passed = all(required_verdicts)
        stage_results.append(
            StageResult(
                stage_id=stage.id,
                stage_title=stage.title,
                passed=passed,
                criteria=criteria_results,
            )
        )

    if not summary:
        passed_count = sum(1 for s in stage_results if s.passed)
        summary = f"Выполнено этапов: {passed_count} из {len(stage_results)}."

    return AnalysisResult(stages=stage_results, summary=summary)
