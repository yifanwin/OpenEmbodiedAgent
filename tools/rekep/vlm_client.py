import base64
import os
import time
import requests


DEFAULT_VLM_MODEL = "gpt-5.4"


def _read_env(name):
    value = os.environ.get(name)
    return value.strip() if isinstance(value, str) else ""


def _read_int_env(name, default=0, lower=0, upper=10):
    raw = _read_env(name)
    if not raw:
        return int(default)
    try:
        value = int(raw)
    except Exception:
        value = int(default)
    return max(int(lower), min(int(upper), value))


def _read_float_env(name, default=1.0, lower=0.0, upper=30.0):
    raw = _read_env(name)
    if not raw:
        return float(default)
    try:
        value = float(raw)
    except Exception:
        value = float(default)
    return max(float(lower), min(float(upper), value))


def resolve_vlm_config(default_model=None):
    api_key = _read_env("REKEP_VLM_API_KEY")
    api_key_env = ""

    if not api_key:
        configured_env = _read_env("REKEP_VLM_API_KEY_ENV")
        if configured_env:
            api_key_env = configured_env
            api_key = _read_env(configured_env)

    if not api_key:
        for fallback_env in ("DMXAPI_API_KEY", "OPENAI_API_KEY"):
            fallback_value = _read_env(fallback_env)
            if fallback_value:
                api_key_env = fallback_env
                api_key = fallback_value
                break

    return {
        "api_key": api_key,
        "api_key_env": api_key_env,
        "base_url": _read_env("REKEP_VLM_BASE_URL"),
        "model": _read_env("REKEP_VLM_MODEL") or default_model or DEFAULT_VLM_MODEL,
    }


def vlm_ready(default_model=None):
    return bool(resolve_vlm_config(default_model=default_model)["api_key"])


def get_vlm_request_config(default_model=None):
    config = resolve_vlm_config(default_model=default_model)
    if not config["api_key"]:
        source = config["api_key_env"] or "REKEP_VLM_API_KEY / REKEP_VLM_API_KEY_ENV / DMXAPI_API_KEY / OPENAI_API_KEY"
        raise RuntimeError(f"VLM API key is required ({source})")
    return config


def encode_image_bytes(image_bytes):
    return base64.b64encode(image_bytes).decode("utf-8")


def build_image_messages(question, image_bytes, system_prompt=None):
    contents = [
        {
            "type": "text",
            "text": question,
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{encode_image_bytes(image_bytes)}",
            },
        },
    ]
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": contents})
    return messages


def _resolve_chat_completions_url(config):
    base_url = config["base_url"] or "https://api.openai.com/v1"
    return f"{base_url.rstrip('/')}/chat/completions"


def _extract_message_text(payload):
    choices = payload.get("choices", [])
    if not choices:
        return ""
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(str(item.get("text", "")))
        return "\n".join(part for part in text_parts if part).strip()
    return str(content).strip()


def request_chat_completion(
    *,
    messages,
    default_model=None,
    temperature=0.0,
    max_tokens=512,
    timeout=300,
):
    config = get_vlm_request_config(default_model=default_model)
    max_retries = _read_int_env("REKEP_VLM_MAX_RETRIES", default=2, lower=0, upper=10)
    retry_backoff_s = _read_float_env("REKEP_VLM_RETRY_BACKOFF_S", default=1.0, lower=0.0, upper=60.0)
    retry_status_codes = {429, 500, 502, 503, 504}
    payload = {
        "model": config["model"],
        "messages": messages,
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    if max_tokens is not None:
        payload["max_tokens"] = int(max_tokens)

    url = _resolve_chat_completions_url(config)
    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }

    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_exc = exc
            if attempt < max_retries:
                sleep_s = retry_backoff_s * (2**attempt)
                if sleep_s > 0:
                    time.sleep(sleep_s)
                continue
            raise RuntimeError(
                f"VLM request failed after {max_retries + 1} attempts "
                f"(model={config['model']}, base_url={config['base_url'] or 'https://api.openai.com/v1'}): {exc}"
            ) from exc

        status_code = int(response.status_code)
        if status_code in retry_status_codes and attempt < max_retries:
            sleep_s = retry_backoff_s * (2**attempt)
            if sleep_s > 0:
                time.sleep(sleep_s)
            continue

        if status_code >= 400:
            body = response.text.strip().replace("\n", " ")
            if len(body) > 400:
                body = f"{body[:400]}...(truncated)"
            raise requests.HTTPError(
                f"{status_code} error from VLM endpoint "
                f"(model={config['model']}, base_url={config['base_url'] or 'https://api.openai.com/v1'}): {body}",
                response=response,
            )

        try:
            return response.json(), config
        except Exception as exc:
            body = response.text.strip().replace("\n", " ")
            if len(body) > 400:
                body = f"{body[:400]}...(truncated)"
            raise RuntimeError(
                f"VLM response JSON parse failed "
                f"(model={config['model']}, base_url={config['base_url'] or 'https://api.openai.com/v1'}): {body}"
            ) from exc

    raise RuntimeError(
        f"VLM request failed after {max_retries + 1} attempts "
        f"(model={config['model']}, base_url={config['base_url'] or 'https://api.openai.com/v1'})"
    ) from last_exc


def ask_image_question(
    *,
    image_bytes,
    question,
    default_model=None,
    system_prompt=None,
    temperature=0.0,
    max_tokens=512,
):
    messages = build_image_messages(question, image_bytes, system_prompt=system_prompt)
    response, config = request_chat_completion(
        messages=messages,
        default_model=default_model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    answer = _extract_message_text(response)
    return answer, config
