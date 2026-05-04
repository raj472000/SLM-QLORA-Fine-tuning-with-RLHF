import re


UNSAFE_PATTERNS = [
    "build a bomb",
    "make malware",
    "steal password",
    "bypass security",
    "phishing",
    "credit card dump",
    "exploit this system"
]


PROMPT_INJECTION_PATTERNS = [
    "ignore previous instructions",
    "reveal system prompt",
    "bypass your rules",
    "developer message",
    "act as unrestricted"
]


PII_PATTERNS = [
    r"\b\d{3}-\d{2}-\d{4}\b",
    r"\b\d{16}\b",
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
]


def contains_prompt_injection(text: str) -> bool:
    text = text.lower()
    return any(pattern in text for pattern in PROMPT_INJECTION_PATTERNS)


def contains_unsafe_intent(text: str) -> bool:
    text = text.lower()
    return any(pattern in text for pattern in UNSAFE_PATTERNS)


def contains_pii(text: str) -> bool:
    return any(re.search(pattern, text) for pattern in PII_PATTERNS)


def validate_prompt(prompt: str) -> tuple[bool, str]:
    if contains_prompt_injection(prompt):
        return False, "Prompt injection detected."

    if contains_unsafe_intent(prompt):
        return False, "Unsafe request detected."

    return True, "safe"


def validate_output(output: str) -> tuple[bool, str]:
    if contains_pii(output):
        return False, "PII detected in output."

    if contains_unsafe_intent(output):
        return False, "Unsafe output detected."

    return True, "safe"