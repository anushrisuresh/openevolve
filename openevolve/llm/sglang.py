"""
SGLang interface for LLMs
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import sglang as sgl

from openevolve.config import LLMModelConfig
from openevolve.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class SGLangLLM(LLMInterface):
    """LLM interface using SGLang"""

    def __init__(
        self,
        model_cfg: LLMModelConfig,
    ):
        self.model = model_cfg.name
        self.system_message = model_cfg.system_message
        self.temperature = model_cfg.temperature
        self.top_p = model_cfg.top_p
        self.max_tokens = model_cfg.max_tokens
        self.timeout = model_cfg.timeout
        self.retries = model_cfg.retries
        self.retry_delay = model_cfg.retry_delay
        self.api_base = model_cfg.api_base
        self.api_key = model_cfg.api_key
        self.random_seed = getattr(model_cfg, "random_seed", None)

        # Set up OpenAI-compatible client for SGLang
        import openai
        self.client = openai.OpenAI(
            base_url=self.api_base,
            api_key=self.api_key or "dummy-key",  # SGLang doesn't require a real API key
        )

        # Only log unique models to reduce duplication
        if not hasattr(logger, "_initialized_models"):
            logger._initialized_models = set()

        if self.model not in logger._initialized_models:
            logger.info(f"Initialized SGLang LLM with model: {self.model}")
            logger._initialized_models.add(self.model)

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text from a prompt"""
        return await self.generate_with_context(
            system_message=self.system_message,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        # Prepare messages with system message
        formatted_messages = [{"role": "system", "content": system_message}]
        formatted_messages.extend(messages)

        # Set up generation parameters
        params = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": kwargs.get("temperature", self.temperature),
            "top_p": kwargs.get("top_p", self.top_p),
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
        }

        # Add seed parameter for reproducibility if configured
        seed = kwargs.get("seed", self.random_seed)
        if seed is not None:
            params["seed"] = seed

        # Attempt the API call with retries
        retries = kwargs.get("retries", self.retries)
        retry_delay = kwargs.get("retry_delay", self.retry_delay)
        timeout = kwargs.get("timeout", self.timeout)

        for attempt in range(retries + 1):
            try:
                # Use asyncio.to_thread to run the synchronous OpenAI client in a thread
                response = await asyncio.wait_for(
                    asyncio.to_thread(self.client.chat.completions.create, **params),
                    timeout=timeout
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt < retries:
                    logger.warning(
                        f"API call failed (attempt {attempt + 1}/{retries + 1}): {e}"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    logger.error(f"API call failed after {retries + 1} attempts: {e}")
                    raise

