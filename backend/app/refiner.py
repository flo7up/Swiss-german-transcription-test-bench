"""Text-model client used to fuse several Realtime hypotheses into one final answer."""

from __future__ import annotations

import os
from typing import Protocol

from .retry import retry_rate_limited
from .transcriber import TokenCache


class TextRefiner(Protocol):
    deployment: str

    async def complete(self, instructions: str, prompt: str) -> str: ...


class AzureOpenAITextRefiner:
    """Call an Azure OpenAI text deployment through the v1 Responses API with Entra ID authentication."""

    _TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"

    def __init__(self, deployment: str, reasoning_effort: str | None = "low", tokens: TokenCache | None = None) -> None:
        self.deployment = deployment
        self.reasoning_effort = reasoning_effort
        self._tokens = tokens or TokenCache()

    @staticmethod
    def _base_url() -> str:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for the ensemble strategy.")
        endpoint = endpoint.rstrip("/")
        if not endpoint.endswith("/openai/v1"):
            endpoint += "/openai/v1"
        return endpoint + "/"

    async def complete(self, instructions: str, prompt: str) -> str:
        from openai import AsyncOpenAI

        token = await self._tokens.get(self._TOKEN_SCOPE)
        options = {"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else {}
        async with AsyncOpenAI(base_url=self._base_url(), api_key=token) as client:
            response = await retry_rate_limited(
                lambda: client.responses.create(
                    model=self.deployment,
                    instructions=instructions,
                    input=prompt,
                    **options,
                )
            )
        text = (response.output_text or "").strip()
        if not text:
            raise RuntimeError(f"Refiner deployment {self.deployment} returned no text.")
        return text
