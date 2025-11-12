import os
from functools import lru_cache
from typing import Generator, List, Dict, Any

from openai import AzureOpenAI, OpenAI, NotFoundError


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

    # New Azure AI Inference endpoints (services.ai.azure.com) expose an OpenAI-compatible API.
    if ".services.ai.azure.com" in endpoint or endpoint.endswith("/openai/v1") or "/openai/" in endpoint:
        return _build_openai_compatible_client(endpoint, key)

    # Classic Azure OpenAI resource (resource.openai.azure.com) requires api-version parameter.
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


def stream_chat(
    deployment: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    deployment_name: str = "AZURE_PHI4_ORCHESTRATOR",
    timeout: float = 60.0,
) -> Generator[str, None, None]:
    model = require_azure_env(deployment, deployment_name)
    client = get_azure_client()
    try:
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            stream=True,
            timeout=timeout,
        )
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
                    yield _content_to_text(delta.content)
    except Exception as exc:
        raise RuntimeError(
            f"Error during Azure OpenAI stream processing: {exc}"
        ) from exc


def complete_chat(
    deployment: str,
    messages: List[Dict[str, str]],
    *,
    temperature: float = 0.2,
    deployment_name: str = "AZURE_PHI4_SPECIALIST",
    timeout: float = 60.0,
) -> str:
    model = require_azure_env(deployment, deployment_name)
    client = get_azure_client()
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            timeout=timeout,
        )
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
    if not resp.choices:
        return ""
    content = resp.choices[0].message.content
    return _content_to_text(content)
