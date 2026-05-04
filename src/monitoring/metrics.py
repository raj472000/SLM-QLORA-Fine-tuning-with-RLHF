from prometheus_client import Counter, Histogram, Gauge


REQUEST_COUNT = Counter(
    "slm_api_requests_total",
    "Total API requests",
    ["endpoint"]
)

FALLBACK_COUNT = Counter(
    "slm_fallback_total",
    "Total fallback events"
)

SAFETY_REJECTION_COUNT = Counter(
    "slm_safety_rejections_total",
    "Total safety rejections"
)

INFERENCE_LATENCY = Histogram(
    "slm_inference_latency_seconds",
    "Inference latency in seconds"
)

GPU_MEMORY_USED = Gauge(
    "slm_gpu_memory_used_mb",
    "GPU memory used in MB"
)