"""Realtime voice sessions via WebSocket.

Connects to the QAI Realtime API for bidirectional audio streaming
with voice activity detection, transcription, and tool calling.

Example::

    import asyncio
    from quantum_sdk import AsyncClient
    from quantum_sdk.realtime import RealtimeConfig, realtime_connect

    async def main():
        client = AsyncClient("qai_key_xxx")
        config = RealtimeConfig(voice="Sal")
        sender, receiver = await realtime_connect(client, config)

        async for event in receiver:
            if event["type"] == "audio_delta":
                pass  # play PCM audio
            elif event["type"] == "transcript_done":
                print(event["transcript"])

        await sender.close()
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import websockets
import websockets.asyncio.client


@dataclass
class RealtimeConfig:
    """Configuration for a realtime voice session."""

    voice: str = "Sal"
    """Voice to use (e.g. 'Sal', 'Eve', 'Vesper')."""

    instructions: str = ""
    """System instructions for the AI."""

    sample_rate: int = 24000
    """PCM sample rate in Hz."""

    tools: list[dict[str, Any]] = field(default_factory=list)
    """Tool definitions (xAI Realtime API format)."""


# Event is a plain dict with a "type" key and event-specific fields.
RealtimeEvent = dict[str, Any]


class RealtimeSender:
    """Write half of a realtime session — send audio and control messages."""

    def __init__(self, ws: websockets.asyncio.client.ClientConnection) -> None:
        self._ws = ws

    async def send_audio(self, base64_pcm: str) -> None:
        """Send a base64-encoded PCM audio chunk."""
        await self._ws.send(json.dumps({
            "type": "input_audio_buffer.append",
            "audio": base64_pcm,
        }))

    async def send_text(self, text: str) -> None:
        """Send a text message and request a response."""
        await self._ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": text}],
            },
        }))
        await self._ws.send(json.dumps({
            "type": "response.create",
            "response": {"modalities": ["text", "audio"]},
        }))

    async def send_function_result(self, call_id: str, output: str) -> None:
        """Send a function/tool call result back to the model."""
        await self._ws.send(json.dumps({
            "type": "conversation.item.create",
            "item": {
                "type": "function_call_output",
                "call_id": call_id,
                "output": output,
            },
        }))
        await self._ws.send(json.dumps({"type": "response.create"}))

    async def cancel_response(self) -> None:
        """Cancel the current response (interrupt)."""
        await self._ws.send(json.dumps({"type": "response.cancel"}))

    async def close(self) -> None:
        """Close the session gracefully."""
        await self._ws.close()


class RealtimeReceiver:
    """Read half of a realtime session — receive audio, transcripts, tool calls."""

    def __init__(self, ws: websockets.asyncio.client.ClientConnection) -> None:
        self._ws = ws

    async def recv(self) -> RealtimeEvent | None:
        """Receive the next event. Returns None when the connection closes."""
        try:
            data = await self._ws.recv()
            if isinstance(data, bytes):
                return None
            return _parse_event(data)
        except websockets.exceptions.ConnectionClosed:
            return None

    def __aiter__(self) -> AsyncIterator[RealtimeEvent]:
        return self

    async def __anext__(self) -> RealtimeEvent:
        event = await self.recv()
        if event is None:
            raise StopAsyncIteration
        return event


async def realtime_connect(
    client: Any,
    config: RealtimeConfig | None = None,
) -> tuple[RealtimeSender, RealtimeReceiver]:
    """Open a realtime voice session via WebSocket.

    Returns (sender, receiver) for bidirectional communication.

    Args:
        client: A quantum_sdk.AsyncClient (or Client — uses its api_key and base_url).
        config: Session configuration. Uses defaults if None.
    """
    if config is None:
        config = RealtimeConfig()

    base_url: str = getattr(client, "base_url", "https://api.quantumencoding.ai")
    api_key: str = getattr(client, "api_key", "")

    # Convert https:// → wss://, http:// → ws://
    ws_base = base_url.replace("https://", "wss://").replace("http://", "ws://")
    url = f"{ws_base}/qai/v1/realtime"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "X-API-Key": api_key,
    }

    ws = await asyncio.wait_for(
        websockets.asyncio.client.connect(url, additional_headers=headers),
        timeout=15.0,
    )

    sender = RealtimeSender(ws)
    receiver = RealtimeReceiver(ws)

    # Send session.update
    session_update = {
        "type": "session.update",
        "session": {
            "voice": config.voice,
            "instructions": config.instructions,
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
            "input_audio_transcription": {"model": "grok-2-audio"},
            "turn_detection": {"type": "server_vad"},
            "tools": config.tools,
        },
    }
    await ws.send(json.dumps(session_update))

    return sender, receiver


def _parse_event(data: str) -> RealtimeEvent:
    """Parse a raw JSON event into a typed RealtimeEvent dict."""
    try:
        v = json.loads(data)
    except json.JSONDecodeError:
        return {"type": "unknown", "raw": data}

    t = v.get("type", "")

    if t == "session.updated":
        return {"type": "session_ready"}

    if t in ("response.audio.delta", "response.output_audio.delta"):
        return {"type": "audio_delta", "delta": v.get("delta", "")}

    if t in ("response.audio_transcript.delta", "response.output_audio_transcript.delta"):
        return {"type": "transcript_delta", "delta": v.get("delta", ""), "source": "output"}

    if t in ("response.audio_transcript.done", "response.output_audio_transcript.done"):
        return {"type": "transcript_done", "transcript": v.get("transcript", ""), "source": "output"}

    if t == "conversation.item.input_audio_transcription.completed":
        return {"type": "transcript_done", "transcript": v.get("transcript", ""), "source": "input"}

    if t == "input_audio_buffer.speech_started":
        return {"type": "speech_started"}

    if t == "input_audio_buffer.speech_stopped":
        return {"type": "speech_stopped"}

    if t == "response.function_call_arguments.done":
        return {
            "type": "function_call",
            "name": v.get("name", ""),
            "call_id": v.get("call_id", ""),
            "arguments": v.get("arguments", ""),
        }

    if t == "response.done":
        return {"type": "response_done"}

    if t == "error":
        err = v.get("error", {})
        msg = err.get("message", "") if isinstance(err, dict) else ""
        if not msg:
            msg = v.get("message", "unknown error")
        return {"type": "error", "message": msg}

    return {"type": "unknown", "raw": v}
