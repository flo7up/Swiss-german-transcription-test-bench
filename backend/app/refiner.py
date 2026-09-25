"""Text-model client used to fuse several Realtime hypotheses into one final answer."""

from __future__ import annotations

from typing import Protocol

from .retry import retry_rate_limited
from .transcriber import TokenCache, azure_openai_base_url


class TextRefiner(Protocol):
    deployment: str

    async def complete(self, instructions: str, prompt: str) -> str: ...


class AzureOpenAITextRefiner:
    """Call an Azure OpenAI text deployment through the v1 Responses API."""

    _TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"

    def __init__(
        self, deployment: str, reasoning_effort: str | None = "low",
        tokens: TokenCache | None = None, api_key: str | None = None,
    ) -> None:
        self.deployment = deployment
        self.reasoning_effort = reasoning_effort
        self._tokens = tokens or TokenCache()
        self._api_key = api_key

    async def complete(self, instructions: str, prompt: str) -> str:
        from openai import AsyncOpenAI

        token = self._api_key or await self._tokens.get(self._TOKEN_SCOPE)
        options = {"reasoning": {"effort": self.reasoning_effort}} if self.reasoning_effort else {}
        async with AsyncOpenAI(base_url=azure_openai_base_url(), api_key=token) as client:
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
