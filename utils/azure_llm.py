import os
import time
from functools import lru_cache
from typing import Generator, List, Dict, Any, Optional

from openai import AzureOpenAI, OpenAI, NotFoundError, BadRequestError
from utils.splunk_logger import get_splunk_logger

logger = get_splunk_logger()

DEFAULT_AZURE_OPENAI_API_VERSION = "2024-02-15-preview"


def require_azure_env(value: str | None, name: str) -> str:
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _build_openai_compatible_client(endpoint: str, key: str) -> OpenAI:
    base = endpoint.rstrip("/")
    if not base.endswith("/openai/v1"):
        base += "/openai/v1/"
    return OpenAI(
        base_url=base,
        api_key=key,
        default_headers={"api-key": key},
    )

@lru_cache(maxsize=1)
def get_azure_client():
    endpoint = require_azure_env(os.getenv("AZURE_OPENAI_ENDPOINT"), "AZURE_OPENAI_ENDPOINT").rstrip("/")
    key = require_azure_env(os.getenv("AZURE_OPENAI_API_KEY"), "AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_OPENAI_API_VERSION)

    if ".services.ai.azure.com" in endpoint or endpoint.endswith("/openai/v1") or "/openai/" in endpoint:
        return _build_openai_compatible_client(endpoint, key)

    return AzureOpenAI(
        api_key=key,
        api_version=api_version,
        azure_endpoint=endpoint,
    )

def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "".join(parts)
    return str(content or "")

def _estimate_tokens(text: str) -> int:
    """Estimate token count using rough approximation of 4 characters per token."""
    if not text:
        return 0
    return max(1, len(text) // 4)

def _estimate_message_tokens(messages: List[Dict[str, str]]) -> int:
    """Estimate total tokens in a list of messages."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        total += _estimate_tokens(_content_to_text(content))
        # Add a few tokens for role and message formatting
        total += 4
    return total

def stream_chat(
    deployment: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    deployment_name: str = "AZURE_PHI4_ORCHESTRATOR",
    timeout: float = 60.0,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> Generator[str, None, None]:
    model = require_azure_env(deployment, deployment_name)
    client = get_azure_client()

    # Start timing and estimate input tokens
    start_time = time.perf_counter()
    tokens_in = _estimate_message_tokens(messages)
    tokens_out = 0
    success = False

    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            timeout=timeout,
        )
    except BadRequestError as exc:
        # Check if this is a content filter violation (jailbreak attempt)
        error_body = getattr(exc, 'body', {}) or {}
        error_detail = error_body.get('error', {})
        error_code = error_detail.get('code', '')

        if error_code == 'content_filter':
            inner_error = error_detail.get('innererror', {})
            filter_result = inner_error.get('content_filter_result', {})

            # Determine filter type and log security event
            is_jailbreak = filter_result.get('jailbreak', {}).get('filtered', False)
            matched_rules = []

            for filter_type in ['jailbreak', 'hate', 'self_harm', 'sexual', 'violence']:
                if filter_result.get(filter_type, {}).get('filtered'):
                    matched_rules.append(f"azure_content_filter_{filter_type}")

            if not matched_rules:
                matched_rules = ["azure_content_filter"]

            # Extract user input preview from messages
            user_input_preview = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_input_preview = _content_to_text(msg.get("content", ""))[:200]
                    break

            logger.log_security_event(
                request_id=request_id or "unknown",
                session_id=session_id or "unknown",
                event_type="content_filter_block",
                blocked=True,
                score=1.0,
                matched_rules=matched_rules,
                user_input_preview=user_input_preview,
                severity="warning" if is_jailbreak else "info"
            )

            # Check what was filtered
            if is_jailbreak:
                raise RuntimeError(
                    "This request was blocked by Azure's content safety filters. "
                    "It appears to be attempting to bypass system instructions or extract sensitive prompts. "
                    "Please rephrase your question to focus on the information you need."
                ) from exc
            else:
                # Other content filter reasons (hate, self-harm, sexual, violence)
                raise RuntimeError(
                    "This request was blocked by Azure's content safety filters. "
                    "Please rephrase your question in a way that complies with content policies."
                ) from exc
        # Re-raise other BadRequestErrors
        raise
    except NotFoundError as exc:
        raise RuntimeError(
            f"Azure OpenAI deployment '{model}' was not found. "
            "Ensure AZURE_PHI4_ORCHESTRATOR (or AZURE_OPENAI_DEPLOYMENT) matches the deployment name in Azure."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Azure OpenAI request timed out after {timeout}s. "
            "The service may be slow or unavailable."
        ) from exc

    try:
        for event in stream:
            for choice in getattr(event, "choices", []):
                delta = getattr(choice, "delta", None)
                if delta and delta.content:
                    content = _content_to_text(delta.content)
                    tokens_out += _estimate_tokens(content)
                    yield content

        success = True
    except Exception as exc:
        raise RuntimeError(
            f"Error during Azure OpenAI stream processing: {exc}"
        ) from exc
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log_llm_call(
            request_id=request_id or "unknown",
            session_id=session_id or "unknown",
            model=model,
            deployment_name=deployment_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            success=success,
            temperature=temperature
        )

def complete_chat(
    deployment: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    deployment_name: str = "AZURE_PHI4_SPECIALIST",
    timeout: float = 60.0,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
) -> str:
    model = require_azure_env(deployment, deployment_name)
    client = get_azure_client()

    # Start timing and estimate input tokens
    start_time = time.perf_counter()
    tokens_in = _estimate_message_tokens(messages)
    tokens_out = 0
    success = False

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
    except BadRequestError as exc:
        # Check if this is a content filter violation (jailbreak attempt)
        error_body = getattr(exc, 'body', {}) or {}
        error_detail = error_body.get('error', {})
        error_code = error_detail.get('code', '')

        if error_code == 'content_filter':
            inner_error = error_detail.get('innererror', {})
            filter_result = inner_error.get('content_filter_result', {})

            # Determine filter type and log security event
            is_jailbreak = filter_result.get('jailbreak', {}).get('filtered', False)
            matched_rules = []

            for filter_type in ['jailbreak', 'hate', 'self_harm', 'sexual', 'violence']:
                if filter_result.get(filter_type, {}).get('filtered'):
                    matched_rules.append(f"azure_content_filter_{filter_type}")

            if not matched_rules:
                matched_rules = ["azure_content_filter"]

            # Extract user input preview from messages
            user_input_preview = ""
            for msg in reversed(messages):
                if msg.get("role") == "user":
                    user_input_preview = _content_to_text(msg.get("content", ""))[:200]
                    break

            logger.log_security_event(
                request_id=request_id or "unknown",
                session_id=session_id or "unknown",
                event_type="content_filter_block",
                blocked=True,
                score=1.0,
                matched_rules=matched_rules,
                user_input_preview=user_input_preview,
                severity="warning" if is_jailbreak else "info"
            )

            # Check what was filtered
            if is_jailbreak:
                raise RuntimeError(
                    "This request was blocked by Azure's content safety filters. "
                    "It appears to be attempting to bypass system instructions or extract sensitive prompts. "
                    "Please rephrase your question to focus on the information you need."
                ) from exc
            else:
                # Other content filter reasons (hate, self-harm, sexual, violence)
                raise RuntimeError(
                    "This request was blocked by Azure's content safety filters. "
                    "Please rephrase your question in a way that complies with content policies."
                ) from exc
        # Re-raise other BadRequestErrors
        raise
    except NotFoundError as exc:
        raise RuntimeError(
            f"Azure OpenAI deployment '{model}' was not found. "
            "Check AZURE_PHI4_EMAIL/AZURE_PHI4_MEETING env vars and ensure the deployments exist."
        ) from exc
    except TimeoutError as exc:
        raise RuntimeError(
            f"Azure OpenAI request timed out after {timeout}s. "
            "The service may be slow or unavailable."
        ) from exc

    try:
        if not resp.choices:
            return ""

        content = resp.choices[0].message.content
        result = _content_to_text(content)
        tokens_out = _estimate_tokens(result)
        success = True

        return result
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.log_llm_call(
            request_id=request_id or "unknown",
            session_id=session_id or "unknown",
            model=model,
            deployment_name=deployment_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            success=success,
            temperature=temperature
        )