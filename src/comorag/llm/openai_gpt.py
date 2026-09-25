import functools
import hashlib
import json
import logging
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import httpx
import openai
from filelock import FileLock
from openai import OpenAI
from openai import AzureOpenAI
from packaging import version
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import (
    TextChatMessage
)

from .base import BaseLLM, LLMConfig

logger = logging.getLogger(__name__)


def cache_response(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # get messages from args or kwargs
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # get model, seed and temperature from kwargs or self.llm_config.generate_params
        gen_params = getattr(self, "llm_config", {}).generate_params if hasattr(self, "llm_config") else {}
        model = kwargs.get("model", gen_params.get("model"))
        seed = kwargs.get("seed", gen_params.get("seed"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))

        # build key data, convert to JSON string and hash to generate key_hash
        key_data = {
            "messages": messages,  # messages requires JSON serializable
            "model": model,
            "seed": seed,
            "temperature": temperature,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # the file name of lock, ensure mutual exclusion when accessing concurrently
        lock_file = self.cache_file_name + ".lock"

        # Try to read from SQLite cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # if the table does not exist, create it
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()  # commit to save the table creation
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                metadata = json.loads(metadata_str)
                # return cached result and mark as hit
                return message, metadata, True

        # if cache miss, call the original function to get the result
        result = func(self, *args, **kwargs)
        message, metadata = result

        # insert new result into cache
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            # make sure the table exists again (if it doesn't exist, it would be created)
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            metadata_str = json.dumps(metadata)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, metadata_str))
            conn.commit()
            conn.close()

        return message, metadata, False

    return wrapper

def dynamic_retry_decorator(func):
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        max_retries = getattr(self, "max_retries", 5)
        dynamic_retry = retry(stop=stop_after_attempt(max_retries), wait=wait_fixed(1))
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

class CacheOpenAI(BaseLLM):
    """OpenAI LLM implementation."""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "CacheOpenAI":
        config_dict = global_config.__dict__
        config_dict['max_retries'] = global_config.max_retry_attempts
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir, global_config, cache_filename: str = None,
                 high_throughput: bool = True,
                 **kwargs) -> None:

        super().__init__()
        self.cache_dir = cache_dir
        self.global_config = global_config

        self.llm_name = global_config.llm_name
        self.llm_base_url = global_config.llm_base_url

        os.makedirs(self.cache_dir, exist_ok=True)
        if cache_filename is None:
            cache_filename = f"{self.llm_name.replace('/', '_')}_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        self._init_llm_config()
        if high_throughput:
            limits = httpx.Limits(max_connections=500, max_keepalive_connections=100)
            client = httpx.Client(limits=limits, timeout=httpx.Timeout(5*60, read=5*60))
        else:
            client = None

        self.max_retries = kwargs.get("max_retries", 2)

        if self.global_config.azure_endpoint is None:
            self.openai_client = OpenAI(base_url=self.llm_base_url, api_key=self.global_config.llm_api_key, http_client=client, max_retries=self.max_retries)
        else:
            self.openai_client = AzureOpenAI(api_version=self.global_config.azure_endpoint.split('api-version=')[1],
                                             azure_endpoint=self.global_config.azure_endpoint, max_retries=self.max_retries)

    def _init_llm_config(self) -> None:
        config_dict = self.global_config.__dict__

        config_dict['llm_name'] = self.global_config.llm_name
        config_dict['llm_base_url'] = self.global_config.llm_base_url

        is_gemini = self.global_config.llm_name.lower().startswith("gemini")

        if is_gemini:
            # Gemini OpenAI-compatible API:
            # use only the parameters that we have verified work.
            config_dict['generate_params'] = {
                "model": self.global_config.llm_name,
                "max_completion_tokens": config_dict.get("max_new_tokens", 400),
                "temperature": config_dict.get("temperature", 0.0),
            }
        else:
            config_dict['generate_params'] = {
                "model": self.global_config.llm_name,
                "max_completion_tokens": config_dict.get("max_new_tokens", 400),
                "n": config_dict.get("num_gen_choices", 1),
                "seed": config_dict.get("seed", 0),
                "temperature": config_dict.get("temperature", 0.0),
            }

        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)


    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        params = deepcopy(self.llm_config.generate_params)
        if kwargs:
            params.update(kwargs)
        params["messages"] = messages


        model_name = params['model'].lower()

        if (
            ('gpt' not in model_name and 'gemini' not in model_name)
            or version.parse(openai.__version__) < version.parse("1.45.0")
        ):
            params['max_tokens'] = params.pop('max_completion_tokens')

        response = self.openai_client.chat.completions.create(
            **params
        )

        # --------------------------------------------------------
        # Validate Gemini's OpenAI-compatible response.
        #
        # HTTP 200 does not guarantee that Gemini returned usable
        # text. Content filters may return a choice with no message.
        # --------------------------------------------------------

        choices = getattr(
            response,
            "choices",
            None,
        )

        if not choices:
            raise RuntimeError(
                "Gemini returned no choices."
            )

        choice = choices[0]

        if choice is None:
            raise RuntimeError(
                "Gemini returned choices[0]=None."
            )

        finish_reason = str(
            getattr(
                choice,
                "finish_reason",
                "",
            )
        )

        finish_reason_upper = (
            finish_reason.upper()
        )

    # --------------------------------------------------------
    # Deterministic/content-policy blocks
    #
    # These should NOT repeatedly crash/retry the experiment.
    # They represent a model/API outcome.
    #
    # <NO_OUTPUT> deliberately contains no A/B/C/D answer.
    # Therefore an MC evaluator should count it as incorrect
    # rather than accidentally treating it as a valid choice.
    # --------------------------------------------------------

        blocked_reasons = (
            "PROHIBITED_CONTENT",
            "SAFETY",
            "BLOCKLIST",
            "SPII",
            "RECITATION",
        )

        is_blocked = any(
            reason in finish_reason_upper
            for reason in blocked_reasons
        )

        usage = getattr(
            response,
            "usage",
            None,
        )

        prompt_tokens = (
            getattr(
                usage,
                "prompt_tokens",
                0,
            )
            if usage is not None
            else 0
        )

        completion_tokens = (
            getattr(
                usage,
                "completion_tokens",
                0,
            )
            if usage is not None
            else 0
        )

        if is_blocked:
            logger.warning(
                "Gemini generation blocked. "
                f"finish_reason={finish_reason}. "
                "Recording <NO_OUTPUT>."
            )

            response_message = (
                "### Final Answer\n"
                "<NO_OUTPUT>"
            )

            metadata = {
                "prompt_tokens": (
                    prompt_tokens
                ),
                "completion_tokens": (
                    completion_tokens
                ),
                "finish_reason": (
                    finish_reason
                ),
                "blocked": True,
            }

            return (
                response_message,
                metadata
            )

    # --------------------------------------------------------
    # Other missing-message cases may be transient.
    # Raise so the existing Tenacity wrapper retries them.
    # --------------------------------------------------------

        message = getattr(
            choice,
            "message",
            None,
        )

        if message is None:
            raise RuntimeError(
                "Gemini returned no message. "
                f"finish_reason={finish_reason}"
            )

        response_message = getattr(
            message,
            "content",
            None,
        )

        if response_message is None:
            raise RuntimeError(
                "Gemini returned content=None. "
                f"finish_reason={finish_reason}"
            )

        if not isinstance(
            response_message,
            str,
        ):
            raise RuntimeError(
                "Gemini returned non-string content. "
                f"type={type(response_message).__name__}, "
                f"finish_reason={finish_reason}"
            )

        response_message = (
            response_message.strip()
        )

        if not response_message:
            raise RuntimeError(
                "Gemini returned empty content. "
                f"finish_reason={finish_reason}"
            )

        metadata = {
            "prompt_tokens": (
                prompt_tokens
            ),
            "completion_tokens": (
                completion_tokens
            ),
            "finish_reason": (
                finish_reason
            ),
            "blocked": False,
        }

        return (
            response_message,
            metadata,
        )
