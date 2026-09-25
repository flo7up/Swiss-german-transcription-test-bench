"""Microsoft Agent Framework adapter for audio-capable Foundry deployments."""

from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
import json
import mimetypes
import os
import time
from typing import Any, Protocol
from urllib.parse import quote

import av
import websockets

from .domain import DatasetItem, ModelDefinition
from .retry import retry_rate_limited
from .settings import BenchmarkSettings


@dataclass(frozen=True)
class TranscriptionResponse:
    transcript: str
    conversation: list[dict[str, Any]]
    time_to_first_token_ms: float | None = None


class AudioTranscriber(Protocol):
    async def transcribe(
        self,
        model: ModelDefinition,
        item: DatasetItem,
        parameters: dict[str, Any],
        prompt: str,
        follow_up_prompt: str | None = None,
    ) -> TranscriptionResponse: ...


class TokenCache:
    """Reuse one Entra token until shortly before expiry instead of invoking the credential chain per clip."""

    _REFRESH_MARGIN_SECONDS = 300

    def __init__(self) -> None:
        self._credential: Any = None
        self._tokens: dict[str, Any] = {}
        self._lock: asyncio.Lock | None = None

    async def get(self, scope: str) -> str:
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            cached = self._tokens.get(scope)
            if cached is not None and cached.expires_on - time.time() > self._REFRESH_MARGIN_SECONDS:
                return cached.token
            if self._credential is None:
                from azure.identity.aio import DefaultAzureCredential

                self._credential = DefaultAzureCredential()
            token = await self._credential.get_token(scope)
            self._tokens[scope] = token
            return token.token


class FoundryAudioTranscriber:
    """Send a multimodal Agent Framework message to a Foundry deployment."""

    def __init__(self, settings: BenchmarkSettings) -> None:
        self.settings = settings

    async def transcribe(
        self,
        model: ModelDefinition,
        item: DatasetItem,
        parameters: dict[str, Any],
        prompt: str,
        follow_up_prompt: str | None = None,
    ) -> TranscriptionResponse:
        if not self.settings.foundry_api_key and not self.settings.foundry_project_endpoint:
            raise RuntimeError("FOUNDRY_PROJECT_ENDPOINT is required for Entra-authenticated Foundry model runs.")
        if not item.audio_path.is_file():
            raise FileNotFoundError(f"Audio file is unavailable: {item.audio_path}")

        media_type = mimetypes.guess_type(item.audio_path.name)[0] or "application/octet-stream"
        if self.settings.foundry_api_key:
            from agent_framework.openai import OpenAIChatClient
            from openai import AsyncOpenAI

            async with AsyncOpenAI(
                base_url=azure_openai_base_url(), api_key=self.settings.foundry_api_key
            ) as openai_client:
                client = OpenAIChatClient(model=model.deployment, async_client=openai_client)
                return await self._run(client, item, parameters, prompt, follow_up_prompt, media_type)

        from agent_framework.foundry import FoundryChatClient
        from azure.identity.aio import DefaultAzureCredential

        credential = DefaultAzureCredential()
        async with credential:
            client = FoundryChatClient(
                project_endpoint=self.settings.foundry_project_endpoint,
                model=model.deployment,
                credential=credential,
            )
            return await self._run(client, item, parameters, prompt, follow_up_prompt, media_type)

    @staticmethod
    async def _run(
        client: Any,
        item: DatasetItem,
        parameters: dict[str, Any],
        prompt: str,
        follow_up_prompt: str | None,
        media_type: str,
    ) -> TranscriptionResponse:
        from agent_framework import Agent, Content, Message

        agent = Agent(
            client=client,
            name="swiss-german-transcriber",
            instructions=(
                "You process Swiss German audio. Follow the user's instructions exactly and return only the "
                "requested text, with no heading, explanation, or confidence statement."
            ),
            default_options=parameters,
        )
        message = Message(
            role="user",
            contents=[
                Content.from_text(text=prompt),
                Content.from_data(
                    data=item.audio_path.read_bytes(),
                    media_type=media_type,
                    additional_properties={"filename": item.audio_path.name},
                ),
            ],
        )
        response = await retry_rate_limited(lambda: agent.run(message))
        conversation = [message.to_dict() for message in response.messages]
        if follow_up_prompt is not None:
            first_pass = response.text.strip()
            follow_up = Message(role="user", contents=[Content.from_text(text=follow_up_prompt)])
            history = [message, *response.messages, follow_up]
            response = await retry_rate_limited(lambda: agent.run(history))
            conversation = [
                {"role": "assistant", "pass": "dialect", "contents": [first_pass]},
                *[entry.to_dict() for entry in response.messages],
            ]

        return TranscriptionResponse(
            transcript=response.text.strip(),
            conversation=conversation,
        )


class RealtimeAudioTranscriber:
    """Run stored audio through an Azure OpenAI Realtime text-response session."""

    _TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
    _PCM_CHUNK_BYTES = 4_800  # 100 ms at 24 kHz, 16-bit mono.

    def __init__(self, settings: BenchmarkSettings, tokens: "TokenCache | None" = None) -> None:
        self.settings = settings
        self._tokens = tokens or TokenCache()
        self._api_key = settings.foundry_api_key if settings else None

    @staticmethod
    def _decode_pcm(audio_path: str) -> bytes:
        """Decode source audio to the 24 kHz PCM format expected by Realtime."""
        resampler = av.AudioResampler(format="s16", layout="mono", rate=24_000)
        chunks: list[bytes] = []
        with av.open(audio_path) as container:
            audio_stream = container.streams.audio[0]
            for frame in container.decode(audio_stream):
                for resampled_frame in resampler.resample(frame):
                    chunks.append(bytes(resampled_frame.planes[0]))
        pcm_audio = b"".join(chunks)
        if not pcm_audio:
            raise ValueError(f"Audio file contains no decodable audio: {audio_path}")
        return pcm_audio

    async def transcribe(
        self,
        model: ModelDefinition,
        item: DatasetItem,
        parameters: dict[str, Any],
        prompt: str,
        follow_up_prompt: str | None = None,
    ) -> TranscriptionResponse:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for realtime model runs.")
        if not item.audio_path.is_file():
            raise FileNotFoundError(f"Audio file is unavailable: {item.audio_path}")

        pcm_audio = self._decode_pcm(str(item.audio_path))
        realtime_endpoint = endpoint.replace("https://", "wss://", 1).rstrip("/")
        websocket_url = f"{realtime_endpoint}/realtime?model={quote(model.deployment, safe='')}"

        token = self._api_key or await self._tokens.get(self._TOKEN_SCOPE)
        transcript, time_to_first_token_ms, first_pass = await retry_rate_limited(
            lambda: self._run_session(websocket_url, token, pcm_audio, prompt, follow_up_prompt)
        )

        conversation: list[dict[str, Any]] = [
            {
                "role": "user",
                "contents": [prompt],
                "audio_file": item.audio_path.name,
                "transport": "azure-openai-realtime",
            }
        ]
        if first_pass is not None:
            conversation.extend(
                [
                    {"role": "assistant", "pass": "dialect", "contents": [first_pass]},
                    {"role": "user", "contents": [follow_up_prompt]},
                ]
            )
        conversation.append({"role": "assistant", "contents": [transcript]})
        return TranscriptionResponse(
            transcript=transcript.strip(),
            time_to_first_token_ms=time_to_first_token_ms,
            conversation=conversation,
        )

    async def _run_session(
        self,
        websocket_url: str,
        token: str,
        pcm_audio: bytes,
        prompt: str,
        follow_up_prompt: str | None = None,
    ) -> tuple[str, float, str | None]:
        async with websockets.connect(
            websocket_url,
            additional_headers={"api-key": token} if self._api_key else {"Authorization": f"Bearer {token}"},
            max_size=2**24,
            open_timeout=20,
        ) as socket:
            await self._receive_event(socket, "session.created")
            await socket.send(
                json.dumps(
                    {
                        "type": "session.update",
                        "session": {
                            "type": "realtime",
                            "output_modalities": ["text"],
                            "audio": {
                                "input": {
                                    "format": {"type": "audio/pcm", "rate": 24_000},
                                    "turn_detection": None,
                                }
                            },
                        },
                    }
                )
            )
            await self._receive_event(socket, "session.updated")

            for offset in range(0, len(pcm_audio), self._PCM_CHUNK_BYTES):
                chunk = pcm_audio[offset : offset + self._PCM_CHUNK_BYTES]
                await socket.send(
                    json.dumps(
                        {
                            "type": "input_audio_buffer.append",
                            "audio": base64.b64encode(chunk).decode("ascii"),
                        }
                    )
                )
            await socket.send(json.dumps({"type": "input_audio_buffer.commit"}))
            response_started = time.perf_counter()
            await socket.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {
                            "output_modalities": ["text"],
                            "instructions": prompt,
                        },
                    }
                )
            )
            first_text, first_token_ms = await self._collect_text_response(socket, response_started)
            if follow_up_prompt is None:
                return first_text, first_token_ms, None

            # The committed audio and first answer stay in the session, so the translation turn can reuse both.
            await socket.send(
                json.dumps(
                    {
                        "type": "conversation.item.create",
                        "item": {
                            "type": "message",
                            "role": "user",
                            "content": [{"type": "input_text", "text": follow_up_prompt}],
                        },
                    }
                )
            )
            await socket.send(
                json.dumps(
                    {
                        "type": "response.create",
                        "response": {"output_modalities": ["text"], "instructions": follow_up_prompt},
                    }
                )
            )
            final_text, final_token_ms = await self._collect_text_response(socket, response_started)
            return final_text, final_token_ms, first_text.strip()
    @staticmethod
    async def _receive_event(socket: Any, expected_type: str) -> dict[str, Any]:
        event = json.loads(await asyncio.wait_for(socket.recv(), timeout=30))
        if event.get("type") == "error":
            raise RuntimeError(event.get("error", {}).get("message", "Realtime API session error."))
        if event.get("type") != expected_type:
            raise RuntimeError(f"Expected Realtime event {expected_type}, received {event.get('type')}.")
        return event

    @staticmethod
    async def _collect_text_response(socket: Any, response_started: float) -> tuple[str, float]:
        transcript = ""
        time_to_first_token_ms = None
        for _ in range(600):
            event = json.loads(await asyncio.wait_for(socket.recv(), timeout=45))
            event_type = event.get("type")
            if event_type in {"response.output_text.delta", "response.text.delta"}:
                delta = event.get("delta", "")
                if delta and time_to_first_token_ms is None:
                    time_to_first_token_ms = (time.perf_counter() - response_started) * 1000
                transcript += delta
                continue
            if event_type == "error":
                raise RuntimeError(event.get("error", {}).get("message", "Realtime API response error."))
            if event_type == "response.done":
                if transcript:
                    if time_to_first_token_ms is None:
                        time_to_first_token_ms = (time.perf_counter() - response_started) * 1000
                    return transcript, time_to_first_token_ms
                for output_item in event.get("response", {}).get("output", []):
                    for content in output_item.get("content", []):
                        transcript += content.get("text") or content.get("transcript") or ""
                if transcript:
                    return transcript, (time.perf_counter() - response_started) * 1000
                raise RuntimeError("Realtime API returned a response without text output.")
        raise RuntimeError("Realtime API response exceeded the expected event limit.")


class RoutedAudioTranscriber:
    """Route a benchmark model to the protocol required by its transport."""

    def __init__(self, settings: BenchmarkSettings) -> None:
        tokens = TokenCache()
        self.foundry = FoundryAudioTranscriber(settings)
        self.realtime = RealtimeAudioTranscriber(settings, tokens)
        self.transcription = TranscriptionApiTranscriber(tokens, settings.foundry_api_key)
        self.audio_chat = AudioChatTranscriber(tokens, settings.foundry_api_key)

    async def transcribe(
        self,
        model: ModelDefinition,
        item: DatasetItem,
        parameters: dict[str, Any],
        prompt: str,
        follow_up_prompt: str | None = None,
    ) -> TranscriptionResponse:
        if model.transport == "azure-openai-realtime":
            return await self.realtime.transcribe(model, item, parameters, prompt, follow_up_prompt)
        if model.transport == TRANSCRIPTION_TRANSPORT:
            return await self.transcription.transcribe(model, item, parameters, prompt, follow_up_prompt)
        if model.transport == "azure-openai-audio-chat":
            return await self.audio_chat.transcribe(model, item, parameters, prompt, follow_up_prompt)
        return await self.foundry.transcribe(model, item, parameters, prompt, follow_up_prompt)


TRANSCRIPTION_TRANSPORT = "azure-openai-transcription"
TRANSCRIPTION_API_VERSION = os.getenv("BENCHMARK_TRANSCRIPTION_API_VERSION", "2025-04-01-preview")
_TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"


def azure_openai_resource_endpoint() -> str:
    """Return the resource root (https://<name>.openai.azure.com) without any /openai path suffix."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for Azure OpenAI model runs.")
    return endpoint.rstrip("/").split("/openai")[0]


def azure_openai_base_url() -> str:
    """Return the Azure OpenAI v1 base URL, accepting endpoints with or without the /openai/v1 suffix."""
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    if not endpoint:
        raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for Azure OpenAI model runs.")
    endpoint = endpoint.rstrip("/")
    if not endpoint.endswith("/openai/v1"):
        endpoint += "/openai/v1"
    return endpoint + "/"


def _audio_media_type(path: Any) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


class TranscriptionApiTranscriber:
    """Send a clip to /audio/transcriptions (gpt-transcribe, gpt-4o-transcribe, whisper, ...).

    Transcription models accept the prompt as context rather than as instructions, so dialect names and
    example sentences steer vocabulary and spelling. They cannot hold a follow-up turn.
    """

    def __init__(self, tokens: TokenCache | None = None, api_key: str | None = None) -> None:
        self._tokens = tokens or TokenCache()
        self._api_key = api_key

    async def transcribe(
        self,
        model: ModelDefinition,
        item: DatasetItem,
        parameters: dict[str, Any],
        prompt: str,
        follow_up_prompt: str | None = None,
    ) -> TranscriptionResponse:
        if follow_up_prompt is not None:
            raise ValueError(f"{model.label} is a transcription model and cannot run a two-pass follow-up turn.")
        if not item.audio_path.is_file():
            raise FileNotFoundError(f"Audio file is unavailable: {item.audio_path}")

        from openai import AsyncAzureOpenAI

        audio = item.audio_path.read_bytes()
        auth = {"api_key": self._api_key} if self._api_key else {"azure_ad_token": await self._tokens.get(_TOKEN_SCOPE)}
        options = {name: value for name, value in parameters.items() if value not in (None, "")}
        # The deployment-scoped route is used because /openai/v1/audio/transcriptions is not routed for every resource.
        async with AsyncAzureOpenAI(
            azure_endpoint=azure_openai_resource_endpoint(),
            api_version=TRANSCRIPTION_API_VERSION,
            **auth,
        ) as client:
            response = await retry_rate_limited(
                lambda: client.audio.transcriptions.create(
                    model=model.deployment,
                    file=(item.audio_path.name, audio, _audio_media_type(item.audio_path)),
                    prompt=prompt,
                    **options,
                )
            )
        transcript = (getattr(response, "text", None) or "").strip()
        if not transcript:
            raise RuntimeError(f"{model.label} returned an empty transcript.")
        return TranscriptionResponse(
            transcript=transcript,
            conversation=[
                {"role": "user", "contents": [prompt], "audio_file": item.audio_path.name, "transport": TRANSCRIPTION_TRANSPORT},
                {"role": "assistant", "contents": [transcript]},
            ],
        )


class AudioChatTranscriber:
    """Send a clip as input_audio to a turn-based audio chat model (gpt-audio family) over Chat Completions."""

    _SYSTEM = (
        "You process Swiss German audio. Follow the user's instructions exactly and return only the requested text, "
        "with no heading, explanation, or confidence statement."
    )

    def __init__(self, tokens: TokenCache | None = None, api_key: str | None = None) -> None:
        self._tokens = tokens or TokenCache()
        self._api_key = api_key

    async def transcribe(
        self,
        model: ModelDefinition,
        item: DatasetItem,
        parameters: dict[str, Any],
        prompt: str,
        follow_up_prompt: str | None = None,
    ) -> TranscriptionResponse:
        if not item.audio_path.is_file():
            raise FileNotFoundError(f"Audio file is unavailable: {item.audio_path}")

        from openai import AsyncOpenAI

        audio_format = item.audio_path.suffix.lstrip(".").lower() or "wav"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self._SYSTEM},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "input_audio",
                        "input_audio": {
                            "data": base64.b64encode(item.audio_path.read_bytes()).decode("ascii"),
                            "format": audio_format,
                        },
                    },
                ],
            },
        ]
        token = self._api_key or await self._tokens.get(_TOKEN_SCOPE)
        options = {name: value for name, value in parameters.items() if value not in (None, "")}
        async with AsyncOpenAI(base_url=azure_openai_base_url(), api_key=token) as client:
            started = time.perf_counter()
            first_text, first_token_ms = await retry_rate_limited(
                lambda: self._stream(client, model.deployment, messages, options, started)
            )
            final_text, final_token_ms = first_text, first_token_ms
            if follow_up_prompt is not None:
                messages = [
                    *messages,
                    {"role": "assistant", "content": first_text},
                    {"role": "user", "content": follow_up_prompt},
                ]
                final_text, final_token_ms = await retry_rate_limited(
                    lambda: self._stream(client, model.deployment, messages, options, started)
                )

        conversation: list[dict[str, Any]] = [
            {"role": "user", "contents": [prompt], "audio_file": item.audio_path.name, "transport": "azure-openai-audio-chat"}
        ]
        if follow_up_prompt is not None:
            conversation += [
                {"role": "assistant", "pass": "dialect", "contents": [first_text]},
                {"role": "user", "contents": [follow_up_prompt]},
            ]
        conversation.append({"role": "assistant", "contents": [final_text]})
        return TranscriptionResponse(transcript=final_text, conversation=conversation, time_to_first_token_ms=final_token_ms)

    @staticmethod
    async def _stream(
        client: Any, deployment: str, messages: list[dict[str, Any]], options: dict[str, Any], started: float
    ) -> tuple[str, float | None]:
        stream = await client.chat.completions.create(model=deployment, messages=messages, stream=True, **options)
        text = ""
        first_token_ms = None
        async for chunk in stream:
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                if first_token_ms is None:
                    first_token_ms = (time.perf_counter() - started) * 1000
                text += delta
        text = text.strip()
        if not text:
            raise RuntimeError(f"Audio chat deployment {deployment} returned no text.")
        return text, first_token_ms