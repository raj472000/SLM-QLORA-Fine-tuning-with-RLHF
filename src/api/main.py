import os
import time
from collections import defaultdict, deque

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.api.schemas import GenerateRequest, GenerateResponse
from src.inference.model_loader import load_model_with_adapter
from src.inference.fallback_router import FallbackRouter
from src.monitoring.metrics import (
    REQUEST_COUNT,
    FALLBACK_COUNT,
    SAFETY_REJECTION_COUNT,
    INFERENCE_LATENCY
)
from src.utils.logger import get_logger

load_dotenv()

logger = get_logger(__name__)

app = FastAPI(
    title="FAANG-Level SLM QLoRA RL API",
    version="1.0.0"
)

MODEL_BASE_NAME = os.getenv(
    "MODEL_BASE_NAME",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

PRIMARY_ADAPTER_PATH = os.getenv(
    "PRIMARY_ADAPTER_PATH",
    "outputs/dpo_adapter"
)

FALLBACK_ADAPTER_PATH = os.getenv(
    "FALLBACK_ADAPTER_PATH",
    "outputs/sft_adapter"
)

RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "30"))

request_history = defaultdict(deque)

router = None


def check_rate_limit(client_ip: str) -> bool:
    now = time.time()
    window_seconds = 60

    queue = request_history[client_ip]

    while queue and now - queue[0] > window_seconds:
        queue.popleft()

    if len(queue) >= RATE_LIMIT_PER_MINUTE:
        return False

    queue.append(now)
    return True


@app.middleware("http")
async def rate_limiter(request: Request, call_next):
    client_ip = request.client.host

    if not check_rate_limit(client_ip):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )

    response = await call_next(request)
    return response


@app.on_event("startup")
def startup():
    global router

    logger.info("Loading primary model")

    primary_model, primary_tokenizer = load_model_with_adapter(
        base_model_name=MODEL_BASE_NAME,
        adapter_path=PRIMARY_ADAPTER_PATH,
        use_4bit=True
    )

    logger.info("Loading fallback model")

    fallback_model, fallback_tokenizer = load_model_with_adapter(
        base_model_name=MODEL_BASE_NAME,
        adapter_path=FALLBACK_ADAPTER_PATH,
        use_4bit=True
    )

    router = FallbackRouter(
        primary_model=primary_model,
        primary_tokenizer=primary_tokenizer,
        fallback_model=fallback_model,
        fallback_tokenizer=fallback_tokenizer
    )


@app.get("/health")
def health():
    REQUEST_COUNT.labels(endpoint="/health").inc()

    return {
        "status": "healthy",
        "model_loaded": router is not None
    }


@app.get("/model-info")
def model_info():
    REQUEST_COUNT.labels(endpoint="/model-info").inc()

    return {
        "base_model": MODEL_BASE_NAME,
        "primary_adapter": PRIMARY_ADAPTER_PATH,
        "fallback_adapter": FALLBACK_ADAPTER_PATH,
        "quantization": "4-bit QLoRA",
        "alignment": "DPO"
    }


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest):
    REQUEST_COUNT.labels(endpoint="/generate").inc()

    with INFERENCE_LATENCY.time():
        result = router.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

    if result["fallback_used"]:
        FALLBACK_COUNT.inc()

    if not result["safe"]:
        SAFETY_REJECTION_COUNT.inc()

    return result


@app.get("/metrics")
def metrics():
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )