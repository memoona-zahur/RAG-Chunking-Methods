"""One ask() path for the LLM - Groq when a key exists, else a dry-run.

Mirrors the week-7 hybrid pattern (`ask_model` doesn't know where the model
runs): the demo pipeline is identical either way. The dry-run is deterministic
and offline: it reports which required facts are present in the provided
context, so the demo stays runnable and honest without any API key.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


def backend() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    return "groq" if key else "dryrun"


def _groq_client():
    from openai import OpenAI

    return OpenAI(api_key=os.getenv("GROQ_API_KEY"), base_url=GROQ_BASE_URL)


def _available_models() -> list[str]:
    try:
        return sorted(m.id for m in _groq_client().models.list().data)
    except Exception as exc:  # noqa: BLE001
        return [f"(could not list models: {exc})"]


def _required_facts_in(context: str) -> list[str]:
    from evaluate import QUESTIONS

    hay = context.lower()
    seen = set()
    for _, facts in QUESTIONS:
        for f in facts:
            if f in hay:
                seen.add(f)
    return sorted(seen)


def ask(messages: list[dict]) -> dict:
    """Single LLM call. Returns dict(reply, input_tokens, output_tokens, backend, model)."""
    if backend() == "groq":
        return _ask_groq(messages)
    return _ask_dryrun(messages)


def ask_robust(messages: list[dict], retries: int = 2) -> dict:
    """Real LLM call when possible; a live failure retries a couple of times,
    then degrades to the deterministic offline path instead of crashing the demo."""
    if backend() != "groq":
        return _ask_dryrun(messages)
    last = None
    for attempt in range(retries + 1):
        try:
            return _ask_groq(messages)
        except Exception as exc:  # noqa: BLE001 -- the demo must never die on a network blip
            last = exc
            if attempt < retries:
                import time

                time.sleep(1.5 * (attempt + 1))
    res = _ask_dryrun(messages)
    res["reply"] = res["reply"].rstrip() + f" (live call failed: {last})"
    res["model"] = "no-llm (dry-run fallback)"
    return res


def _ask_groq(messages: list[dict]) -> dict:
    from openai import NotFoundError

    client = _groq_client()
    try:
        resp = client.chat.completions.create(model=GROQ_MODEL, messages=messages)
    except NotFoundError as exc:
        raise RuntimeError(
            f"GROQ_MODEL='{GROQ_MODEL}' is not available on your account. "
            f"Set GROQ_MODEL in .env to one of: {', '.join(_available_models())}"
        ) from exc
    usage = resp.usage
    return {
        "reply": resp.choices[0].message.content,
        "input_tokens": usage.prompt_tokens if usage else 0,
        "output_tokens": usage.completion_tokens if usage else 0,
        "backend": "groq",
        "model": GROQ_MODEL,
    }


def _ask_dryrun(messages: list[dict]) -> dict:
    """Deterministic offline answer used when no key is set (or a live call failed)."""
    context = "\n".join(m.get("content", "") for m in messages if m.get("role") == "user")
    facts = _required_facts_in(context)
    reply = (
        f"[DRY-RUN - no GROQ key set] Context read holds required facts "
        f"{facts if facts else 'none of the four test questions'}. "
        f"With a live LLM this context would be used to answer. "
    )
    return {
        "reply": reply,
        "input_tokens": len(context) // 4,
        "output_tokens": len(reply) // 4,
        "backend": "dryrun",
        "model": "no-llm (dry-run)",
    }


def describe_backend() -> str:
    return f"LLM backend: {backend()} ({GROQ_MODEL if backend() == 'groq' else 'no key -> dry-run'})"