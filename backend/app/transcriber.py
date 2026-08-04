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
    ) -> TranscriptionResponse: ...


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
    ) -> TranscriptionResponse:
        if not self.settings.foundry_project_endpoint:
            raise RuntimeError("FOUNDRY_PROJECT_ENDPOINT is required before a model run can start.")
        if not item.audio_path.is_file():
            raise FileNotFoundError(f"Audio file is unavailable: {item.audio_path}")

        from agent_framework import Agent, Content, Message
        from agent_framework.foundry import FoundryChatClient
        from azure.identity.aio import DefaultAzureCredential

        media_type = mimetypes.guess_type(item.audio_path.name)[0] or "application/octet-stream"
        credential = DefaultAzureCredential()
        async with credential:
            client = FoundryChatClient(
                project_endpoint=self.settings.foundry_project_endpoint,
                model=model.deployment,
                credential=credential,
            )
            agent = Agent(
                client=client,
                name="swiss-german-transcriber",
                instructions=(
                    "You transcribe Swiss German audio faithfully. Return only the transcript, with no heading, "
                    "translation, explanation, or confidence statement. Preserve dialectal spellings when audible."
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

        return TranscriptionResponse(
            transcript=response.text.strip(),
            conversation=[message.to_dict() for message in response.messages],
        )


class RealtimeAudioTranscriber:
    """Run stored audio through an Azure OpenAI Realtime text-response session."""

    _TOKEN_SCOPE = "https://cognitiveservices.azure.com/.default"
    _PCM_CHUNK_BYTES = 4_800  # 100 ms at 24 kHz, 16-bit mono.

    def __init__(self, settings: BenchmarkSettings) -> None:
        self.settings = settings

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
    ) -> TranscriptionResponse:
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if not endpoint:
            raise RuntimeError("AZURE_OPENAI_ENDPOINT is required for realtime model runs.")
        if not item.audio_path.is_file():
            raise FileNotFoundError(f"Audio file is unavailable: {item.audio_path}")

        pcm_audio = self._decode_pcm(str(item.audio_path))
        realtime_endpoint = endpoint.replace("https://", "wss://", 1).rstrip("/")
        websocket_url = f"{realtime_endpoint}/realtime?model={quote(model.deployment, safe='')}"

        from azure.identity.aio import DefaultAzureCredential

        credential = DefaultAzureCredential()
        try:
            token = (await credential.get_token(self._TOKEN_SCOPE)).token
            transcript, time_to_first_token_ms = await retry_rate_limited(
                lambda: self._run_session(websocket_url, token, pcm_audio, prompt)
            )
        finally:
            await credential.close()

        return TranscriptionResponse(
            transcript=transcript.strip(),
            time_to_first_token_ms=time_to_first_token_ms,
            conversation=[
                {
                    "role": "user",
                    "contents": [prompt],
                    "audio_file": item.audio_path.name,
                    "transport": "azure-openai-realtime",
                },
                {"role": "assistant", "contents": [transcript]},
            ],
        )

    async def _run_session(
        self, websocket_url: str, token: str, pcm_audio: bytes, prompt: str
    ) -> tuple[str, float]:
        async with websockets.connect(
            websocket_url,
            additional_headers={"Authorization": f"Bearer {token}"},
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
            return await self._collect_text_response(socket, response_started)

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
        for _ in range(200):
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
        self.foundry = FoundryAudioTranscriber(settings)
        self.realtime = RealtimeAudioTranscriber(settings)

    async def transcribe(
        self,
        model: ModelDefinition,
        item: DatasetItem,
        parameters: dict[str, Any],
        prompt: str,
    ) -> TranscriptionResponse:
        if model.transport == "azure-openai-realtime":
            return await self.realtime.transcribe(model, item, parameters, prompt)
        return await self.foundry.transcribe(model, item, parameters, prompt)