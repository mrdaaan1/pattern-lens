from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.analyzer import AnalysisResult, analyze_transcript
from app.llm_providers import DEFAULT_MODEL, DEFAULT_PROVIDER, LLMConfigError, PROVIDERS
from app.stages import STAGES, Stage

app = FastAPI(title="Pattern Lens")


class AnalyzeRequest(BaseModel):
    transcript: str
    provider: str = DEFAULT_PROVIDER
    model: str = DEFAULT_MODEL


@app.get("/api/v1/stages")
async def api_list_stages() -> list[Stage]:
    return STAGES


@app.get("/api/v1/llm-providers")
async def api_llm_providers() -> dict[str, list[str]]:
    return PROVIDERS


@app.post("/api/v1/analyze")
async def api_analyze(request: AnalyzeRequest) -> AnalysisResult:
    if not request.transcript.strip():
        raise HTTPException(status_code=400, detail="Транскрибация пуста")

    try:
        result = await analyze_transcript(request.transcript, request.provider, request.model)
    except LLMConfigError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")


app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
