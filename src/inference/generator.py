import time
import torch

from src.inference.safety import validate_prompt, validate_output


def format_prompt(tokenizer, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": prompt
        }
    ]

    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
    except Exception:
        return f"### Instruction:\n{prompt}\n\n### Response:\n"


def generate_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9
) -> dict:
    safe, reason = validate_prompt(prompt)

    if not safe:
        return {
            "output": "I cannot help with that request.",
            "safe": False,
            "reason": reason,
            "fallback_used": True,
            "latency_ms": 0
        }

    formatted_prompt = format_prompt(tokenizer, prompt)

    start = time.time()

    inputs = tokenizer(
        formatted_prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    latency_ms = round((time.time() - start) * 1000, 2)

    output_safe, output_reason = validate_output(output)

    if not output_safe:
        return {
            "output": "The generated answer was blocked by safety filters.",
            "safe": False,
            "reason": output_reason,
            "fallback_used": True,
            "latency_ms": latency_ms
        }

    return {
        "output": output,
        "safe": True,
        "reason": "safe",
        "fallback_used": False,
        "latency_ms": latency_ms
    }