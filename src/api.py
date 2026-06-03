import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from src.agent import ConversationalAgent


def _classify_agent_error(exc: Exception):
    """
    Turn an agent/LLM exception into a clean HTTP error. Gemini's free tier
    raises RESOURCE_EXHAUSTED (HTTP 429) when a request quota is hit — either
    the per-minute rate limit or the per-day cap. Surface that as a 429 with a
    clear message (and distinguish the two when the error says so) instead of a
    bare 500.
    """
    msg = str(exc)
    if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
        per_day = "PerDay" in msg or "free_tier_requests" in msg
        if per_day:
            detail = (
                "Gemini free-tier DAILY quota exhausted for this model "
                "(each chat message is ~2 requests). This does NOT reset in a "
                "minute — wait for the daily reset (~midnight US-Pacific) or "
                "use a different API key / Google account for a fresh quota."
            )
        else:
            detail = (
                "Gemini free-tier rate limit reached (a few requests/minute; "
                "each chat message is ~2 requests). Wait ~60s and try again."
            )
        return HTTPException(status_code=429, detail=detail)
    logging.exception("Agent error")
    return HTTPException(status_code=500, detail=f"Agent error: {msg}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Single shared agent instance, populated in the lifespan handler on startup.
_agent: ConversationalAgent | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the agent once when the server starts; release it on shutdown."""
    global _agent
    logging.info("Starting up: building conversational agent...")
    _agent = ConversationalAgent().build()
    logging.info("Agent ready. API is live.")
    yield
    _agent = None
    logging.info("Shutting down.")


app = FastAPI(
    title="Adult Income Conversational Agent",
    description=(
        "A domain-aware agent (HW2) that answers US-census income questions "
        "via RAG and makes income predictions with the Homework 1 model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# Schemas
class ChatRequest(BaseModel):
    """Request body for /chat and /chat/stream."""
    message: str = Field(
        ...,
        description="The user's natural-language message.",
        examples=["Would a 45-year-old with a Bachelors degree earn over $50K?"],
    )
    session_id: str = Field(
        ...,
        description="Conversation id. Reuse it across turns to keep memory.",
        examples=["user_001"],
    )


class ChatResponse(BaseModel):
    """Response body for /chat."""
    response: str = Field(..., description="The agent's reply.")


# Endpoints
@app.get("/health", tags=["meta"])
def health():
    """Liveness probe: reports whether the agent finished building."""
    return {"status": "ok", "agent_ready": _agent is not None}


@app.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(request: ChatRequest):
    """
    Send one message to the agent and get the complete reply.

    The agent autonomously decides whether to retrieve domain knowledge, run an
    income prediction, query dataset statistics, or just answer directly.
    """
    try:
        reply = _agent.chat(request.message, session_id=request.session_id)
    except Exception as exc:
        raise _classify_agent_error(exc)
    return ChatResponse(response=reply)


@app.post("/chat/stream", tags=["chat"])
async def chat_stream(request: ChatRequest):
    """
    Same as /chat, but streams the reply token-by-token via Server-Sent Events
    (bonus Task 6). Each SSE `data:` frame carries one text chunk; a final
    `event: done` frame signals completion.
    """
    async def event_generator():
        try:
            async for token in _agent.astream_chat(
                request.message, session_id=request.session_id
            ):
                yield {"event": "message", "data": token}
        except Exception as exc:
            # The response has already started, so we can't set an HTTP status;
            # surface the problem as a dedicated SSE error event instead.
            yield {"event": "error", "data": _classify_agent_error(exc).detail}
        yield {"event": "done", "data": "[DONE]"}

    return EventSourceResponse(event_generator())
