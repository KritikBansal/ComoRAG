import logging
import os
from abc import ABC, abstractmethod

from openai import OpenAI
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)


class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self, context, max_tokens=150):
        pass
class ProhibitedContentError(RuntimeError):
    """
    Raised when Gemini deterministically blocks a prompt
    because of prohibited content.
    """
    pass

class GPT4SummarizationModel(BaseSummarizationModel):
    def __init__(self, model=None, llm_base_url="https://api.example.com/v1",llm_api_key="your-api-key-here"):
        """
        Initialize class with support for custom model and API base URL.

        :param model: Model name
        :param llm_base_url: Base URL for LLM API
        """
        self.model = model
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        # Set up OpenAI client with custom base URL
        self.client = OpenAI(base_url=self.llm_base_url,api_key=self.llm_api_key,)

    @retry(
        retry=retry_if_not_exception_type(
            ProhibitedContentError
        ),
        wait=wait_random_exponential(
            min=2,
            max=30,
        ),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def summarize(
        self,
        context,
        max_completion_tokens=500,
        stop_sequence=None,
    ):
        """
        Generate a summary using the configured LLM.

        Invalid or incomplete API responses are treated as
        transient failures and retried.
        """

        request_params = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant.",
                },
                {
                    "role": "user",
                    "content": (
                        "Write a summary of the following, "
                        "including as many key details as possible: "
                        f"{context}"
                    ),
                },
            ],
            "max_completion_tokens": max_completion_tokens,
            "temperature": 0,
            "top_p": 1,
        }

        if stop_sequence is not None:
            request_params["stop"] = stop_sequence

        response = self.client.chat.completions.create(
            **request_params
        )

        # --------------------------------------------------------
        # Validate the OpenAI-compatible Gemini response.
        # A HTTP 200 does not necessarily mean that a usable
        # completion was returned.
        # --------------------------------------------------------
        choices = getattr(
            response,
            "choices",
            None,
        )

        if not choices:
            raise RuntimeError(
                "Summarization API returned no choices."
            )

        choice = choices[0]

        if choice is None:
            raise RuntimeError(
                "Summarization API returned choices[0]=None."
            )

        finish_reason = str(
            getattr(
                choice,
                "finish_reason",
                "",
            )
        ).upper()

        if "PROHIBITED_CONTENT" in finish_reason:
            raise ProhibitedContentError(
                "Gemini blocked summarization with "
                f"finish_reason={finish_reason}"
            )

        message = getattr(
            choice,
            "message",
            None,
        )

        if message is None:
            raise RuntimeError(
                "Summarization API returned no message."
            )

        content = getattr(
            message,
            "content",
            None,
        )

        if content is None:
            raise RuntimeError(
                "Summarization API returned content=None."
            )

        if not isinstance(content, str):
            raise RuntimeError(
                "Summarization API returned non-string content: "
                f"{type(content).__name__}"
            )

        content = content.strip()

        if not content:
            raise RuntimeError(
                "Summarization API returned empty content."
            )

        return content
