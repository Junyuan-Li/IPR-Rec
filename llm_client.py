import json
import os
import re
import threading
import urllib.error
import urllib.request


_DOTENV_LOCK = threading.Lock()
_DOTENV_LOADED = False
_DOTENV_VALUES = {}
_QUOTE_CHARS = {'"', "'", '“', '”', '‘', '’', '＂', '＇'}


def _strip_env_value(raw_value: str) -> str:
    value = raw_value.strip().strip('\ufeff').strip('\u200b').strip('\u3000')
    while len(value) >= 2 and value[0] == value[-1] and value[0] in _QUOTE_CHARS:
        value = value[1:-1].strip().strip('\ufeff').strip('\u200b').strip('\u3000')
    return value


def _sanitize_api_key(raw_value: str) -> str:
    api_key = _strip_env_value(raw_value)
    api_key = api_key.strip(''.join(_QUOTE_CHARS)).strip()
    return api_key


def _get_env_setting(*keys: str, default: str = '') -> str:
    for key in keys:
        value = _DOTENV_VALUES.get(key)
        if value:
            return _strip_env_value(value)
        value = os.getenv(key)
        if value:
            return _strip_env_value(value)
    return default


def _iter_parent_dirs(start_dir: str):
    current_dir = os.path.abspath(start_dir)
    visited = set()
    while current_dir and current_dir not in visited:
        visited.add(current_dir)
        yield current_dir
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir


def _iter_env_candidates():
    visited_paths = set()
    search_roots = [os.getcwd(), os.path.dirname(os.path.abspath(__file__))]
    for root_dir in search_roots:
        if not root_dir:
            continue
        for current_dir in _iter_parent_dirs(root_dir):
            env_path = os.path.join(current_dir, '.env')
            if env_path in visited_paths:
                continue
            visited_paths.add(env_path)
            yield env_path


def _load_local_dotenv() -> None:
    global _DOTENV_LOADED
    if _DOTENV_LOADED:
        return
    with _DOTENV_LOCK:
        if _DOTENV_LOADED:
            return
        for env_path in _iter_env_candidates():
            if not os.path.exists(env_path):
                continue
            with open(env_path, 'r', encoding='utf-8') as env_file:
                for raw_line in env_file:
                    line = raw_line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    key, value = line.split('=', 1)
                    key = key.strip()
                    if not key:
                        continue
                    sanitized_value = _strip_env_value(value)
                    _DOTENV_VALUES[key] = sanitized_value
                    os.environ.setdefault(key, sanitized_value)
            break
        _DOTENV_LOADED = True


_load_local_dotenv()


DEFAULT_SYSTEM_PROMPT = 'You are a path planning agent for recommendation.'
DEFAULT_QWEN_BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions'
DEFAULT_QWEN_MODEL = 'qwen-turbo'
DEFAULT_BACKEND = 'dashscope'
LLM_CACHE_ENABLED = _get_env_setting('ONEREC_LLM_CACHE', default='1').strip().lower() not in {'0', 'false', 'no'}
_LLM_RESPONSE_CACHE = {}
_LLM_CACHE_LOCK = threading.Lock()


def _get_api_key() -> str:
    raw_candidates = [
        _DOTENV_VALUES.get('QWEN_API_KEY'),
        _DOTENV_VALUES.get('DASHSCOPE_API_KEY'),
        os.getenv('QWEN_API_KEY'),
        os.getenv('DASHSCOPE_API_KEY'),
    ]
    api_key = ''
    last_error = ''
    for raw_value in raw_candidates:
        if not raw_value:
            continue
        candidate = _sanitize_api_key(raw_value)
        if not candidate:
            last_error = 'LLM API key is empty after sanitization. Check your local .env or environment variable formatting.'
            continue
        if any(ord(ch) > 127 for ch in candidate):
            last_error = 'LLM API key contains non-ASCII characters after sanitization. Remove smart quotes or full-width characters from QWEN_API_KEY / DASHSCOPE_API_KEY.'
            continue
        api_key = candidate
        break
    if not api_key:
        if last_error:
            raise RuntimeError(last_error)
        raise RuntimeError('Set QWEN_API_KEY or DASHSCOPE_API_KEY in environment variables or a local .env file before calling the LLM.')
    return api_key


def validate_llm_configuration() -> dict:
    backend = _strip_env_value(_get_env_setting('ONEREC_LLM_BACKEND', default=DEFAULT_BACKEND)).strip().lower()
    if backend == 'local':
        return {
            'ok': True,
            'backend': backend,
            'base_url': 'local',
            'model': 'local',
            'error': '',
        }
    try:
        _ = _get_api_key()
        base_url = _get_env_setting('QWEN_BASE_URL', 'DASHSCOPE_BASE_URL', default=DEFAULT_QWEN_BASE_URL)
        model_name = _get_env_setting('QWEN_MODEL', 'DASHSCOPE_MODEL', default=DEFAULT_QWEN_MODEL)
        return {
            'ok': True,
            'backend': backend,
            'base_url': base_url,
            'model': model_name,
            'error': '',
        }
    except Exception as exc:
        return {
            'ok': False,
            'backend': backend,
            'base_url': _get_env_setting('QWEN_BASE_URL', 'DASHSCOPE_BASE_URL', default=DEFAULT_QWEN_BASE_URL),
            'model': _get_env_setting('QWEN_MODEL', 'DASHSCOPE_MODEL', default=DEFAULT_QWEN_MODEL),
            'error': str(exc).strip(),
        }


def call_llm(prompt: str) -> str:
    api_key = _get_api_key()
    base_url = _get_env_setting('QWEN_BASE_URL', 'DASHSCOPE_BASE_URL', default=DEFAULT_QWEN_BASE_URL)
    model_name = _get_env_setting('QWEN_MODEL', 'DASHSCOPE_MODEL', default=DEFAULT_QWEN_MODEL)
    temperature = float(_get_env_setting('QWEN_TEMPERATURE', 'DASHSCOPE_TEMPERATURE', default='0.0'))
    timeout = float(_get_env_setting('QWEN_TIMEOUT', 'DASHSCOPE_TIMEOUT', default='60'))
    cache_key = (base_url, model_name, round(temperature, 6), prompt)

    if LLM_CACHE_ENABLED:
        with _LLM_CACHE_LOCK:
            cached_response = _LLM_RESPONSE_CACHE.get(cache_key)
        if cached_response is not None:
            return cached_response

    payload = {
        'model': model_name,
        'messages': [
            {'role': 'system', 'content': DEFAULT_SYSTEM_PROMPT},
            {'role': 'user', 'content': prompt},
        ],
        'temperature': temperature,
    }
    request = urllib.request.Request(
        url=base_url,
        data=json.dumps(payload).encode('utf-8'),
        headers={
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
        },
        method='POST',
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as exc:
        error_body = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'Qwen API request failed: status={exc.code} body={error_body}') from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Qwen API request failed: {exc.reason}') from exc

    choices = response_data.get('choices') or []
    if not choices:
        raise RuntimeError(f'Qwen API returned no choices: {response_data}')

    message = choices[0].get('message') or {}
    content = message.get('content')
    if not content:
        raise RuntimeError(f'Qwen API returned empty content: {response_data}')
    content = content.strip()
    if LLM_CACHE_ENABLED:
        with _LLM_CACHE_LOCK:
            _LLM_RESPONSE_CACHE[cache_key] = content
    return content


class DashScopePlannerEngine:
    def build_prompt(self, history_titles, candidate_title: str) -> str:
        history_text = '; '.join(history_titles[-5:]) if history_titles else 'N/A'
        prompt = f"""
You are simulating a news reader.

User reading history:
{history_text}

Candidate news:
{candidate_title}

Question:
Will the user click this news?

Answer only YES or NO.
"""
        return prompt.strip()

    def generate(self, prompt: str, max_tokens: int = 10) -> str:
        del max_tokens
        return call_llm(prompt)

    def plan_path(self, prompt: str, max_tokens: int = 50) -> str:
        del max_tokens
        return call_llm(prompt)

    def parse_path(self, response: str, candidate_ids: list):
        indices = list(map(int, re.findall(r'\d+', response)))
        path = []
        for idx in indices:
            if 0 <= idx < len(candidate_ids):
                path.append(candidate_ids[idx])
        return path


def get_planner_engine(backend: str = None):
    selected_backend = _strip_env_value(backend) if backend else _get_env_setting('ONEREC_LLM_BACKEND', default=DEFAULT_BACKEND)
    selected_backend = selected_backend.strip().lower()
    if selected_backend in {'dashscope', 'api', 'remote'}:
        return DashScopePlannerEngine()
    if selected_backend == 'local':
        from qwen_local_feedback import get_feedback_engine

        return get_feedback_engine()
    raise ValueError(f'Unsupported ONEREC_LLM_BACKEND: {selected_backend}')