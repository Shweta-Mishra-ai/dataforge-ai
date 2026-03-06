import logging
from groq import Groq
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from config.settings import config

logger = logging.getLogger(__name__)


class LLMClient:

    def __init__(self, api_key: str):
        self._client = Groq(api_key=api_key)
        self.model   = config.llm_model

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    def chat(self, messages: list, system: str = "") -> str:
        full = []
        if system:
            full.append({"role": "system", "content": system})
        full.extend(messages)

        resp = self._client.chat.completions.create(
            messages=full,
            model=self.model,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens,
            timeout=config.llm_timeout_sec,
        )
        return resp.choices[0].message.content

    def chat_safe(
        self,
        messages: list,
        system: str = "",
        fallback: str = '{"tool":"none","params":{},"explanation":"Unable to process."}'
    ) -> str:
        try:
            return self.chat(messages, system)
        except Exception as e:
            logger.error(f"LLM failed: {e}")
            return fallback
