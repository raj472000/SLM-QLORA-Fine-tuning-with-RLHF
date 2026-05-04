from src.inference.generator import generate_text
from src.utils.logger import get_logger

logger = get_logger(__name__)


class FallbackRouter:
    def __init__(self, primary_model, primary_tokenizer, fallback_model=None, fallback_tokenizer=None):
        self.primary_model = primary_model
        self.primary_tokenizer = primary_tokenizer
        self.fallback_model = fallback_model
        self.fallback_tokenizer = fallback_tokenizer

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, top_p: float) -> dict:
        try:
            result = generate_text(
                model=self.primary_model,
                tokenizer=self.primary_tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )

            if result["safe"]:
                return result

            logger.warning("Primary model response blocked. Trying fallback model.")

        except Exception as exc:
            logger.error(f"Primary model failed: {exc}")

        if self.fallback_model and self.fallback_tokenizer:
            try:
                fallback_result = generate_text(
                    model=self.fallback_model,
                    tokenizer=self.fallback_tokenizer,
                    prompt=prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p
                )

                fallback_result["fallback_used"] = True
                return fallback_result

            except Exception as exc:
                logger.error(f"Fallback model failed: {exc}")

        return {
            "output": "The system could not safely generate a response.",
            "safe": False,
            "reason": "primary_and_fallback_failed",
            "fallback_used": True,
            "latency_ms": 0
        }