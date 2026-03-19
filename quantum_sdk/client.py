"""Quantum AI API client — sync and async."""

from __future__ import annotations

import json
import time
from typing import Any, Iterator, AsyncIterator

import httpx

from .errors import APIError
from .types import (
    ChatRequest,
    ChatResponse,
    ChatUsage,
    ContentBlock,
    StreamDelta,
    StreamEvent,
    StreamToolUse,
    ImageRequest,
    ImageResponse,
    ImageEditRequest,
    ImageEditResponse,
    VideoRequest,
    VideoResponse,
    TTSRequest,
    TTSResponse,
    STTRequest,
    STTResponse,
    MusicRequest,
    MusicResponse,
    SoundEffectResponse,
    EmbedRequest,
    EmbedResponse,
    DocumentRequest,
    DocumentResponse,
    RAGSearchRequest,
    RAGSearchResponse,
    RAGCorpus,
    SurrealRAGSearchRequest,
    SurrealRAGSearchResponse,
    SurrealRAGProvidersResponse,
    ModelInfo,
    PricingInfo,
    BalanceResponse,
    UsageEntry,
    UsageResponse,
    UsageSummaryResponse,
    PricingEntry,
    PricingResponse,
    JobCreateResponse,
    JobStatusResponse,
    JobListResponse,
    SessionChatRequest,
    SessionChatResponse,
    ContextConfig,
    AgentRunRequest,
    AgentWorker,
    MissionRunRequest,
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyListResponse,
    APIKeyInfo,
    ComputeTemplate,
    ComputeProvisionRequest,
    ComputeProvisionResponse,
    ComputeInstance,
    VoiceListResponse,
    VoiceCloneResponse,
    VoiceInfo,
    DialogueRequest,
    DialogueVoice,
    AudioResponse,
    AlignmentResponse,
    VoiceDesignResponse,
    VideoStudioRequest,
    VideoTranslateRequest,
    VideoPhotoAvatarRequest,
    VideoDigitalTwinRequest,
    HeyGenAvatar,
    HeyGenVoice,
    HeyGenTemplate,
    ChunkDocumentRequest,
    ChunkDocumentResponse,
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    ContactRequest,
    ContactResponse,
    ChatTool,
)

DEFAULT_BASE_URL = "https://api.quantumencoding.ai"
DEFAULT_TIMEOUT = 60.0


class _ResponseMeta:
    """Parsed response metadata from HTTP headers."""

    __slots__ = ("cost_ticks", "request_id", "model")

    def __init__(self, headers: httpx.Headers) -> None:
        self.request_id = headers.get("x-qai-request-id", "")
        self.model = headers.get("x-qai-model", "")
        raw_ticks = headers.get("x-qai-cost-ticks", "")
        self.cost_ticks = int(raw_ticks) if raw_ticks else 0


def _parse_api_error(response: httpx.Response, request_id: str) -> APIError:
    """Parse an error response body into an APIError."""
    body = response.text
    code = str(response.status_code)
    message = body

    try:
        data = response.json()
        err_obj = data.get("error", {})
        if isinstance(err_obj, dict):
            if err_obj.get("message"):
                message = err_obj["message"]
            code = err_obj.get("code") or err_obj.get("type") or code
    except Exception:
        pass

    return APIError(
        status_code=response.status_code,
        code=code,
        message=message,
        request_id=request_id or None,
    )


def _parse_sse_event(payload: str) -> StreamEvent:
    """Parse a single SSE JSON payload into a StreamEvent."""
    raw: dict[str, Any] = json.loads(payload)
    event_type: str = raw.get("type", "")
    ev = StreamEvent(type=event_type)

    if event_type in ("content_delta", "thinking_delta"):
        delta_data = raw.get("delta")
        if isinstance(delta_data, dict):
            ev.delta = StreamDelta(text=delta_data.get("text", ""))
    elif event_type == "tool_use":
        ev.tool_use = StreamToolUse(
            id=raw.get("id", ""),
            name=raw.get("name", ""),
            input=raw.get("input", {}),
        )
    elif event_type == "usage":
        ev.usage = ChatUsage(
            input_tokens=raw.get("input_tokens", 0),
            output_tokens=raw.get("output_tokens", 0),
            cost_ticks=raw.get("cost_ticks", 0),
        )
    elif event_type == "error":
        ev.error = raw.get("message", "")

    return ev


def _iter_sse_lines(lines_iter: Iterator[str]) -> Iterator[StreamEvent]:
    """Shared SSE line parser for sync iterators."""
    for line in lines_iter:
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            yield StreamEvent(type="done", done=True)
            return
        try:
            yield _parse_sse_event(payload)
        except json.JSONDecodeError as exc:
            yield StreamEvent(type="error", error=f"parse SSE: {exc}")
            return


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------


class Client:
    """Synchronous Quantum AI API client."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.Client | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.Client(timeout=timeout)
        self._owns_client = http_client is None

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def _do_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[Any, _ResponseMeta]:
        url = self.base_url + path
        headers = self._headers()
        if body is not None:
            headers["Content-Type"] = "application/json"

        resp = self._client.request(
            method,
            url,
            headers=headers,
            content=json.dumps(body).encode() if body is not None else None,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    def _do_json_no_auth(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[Any, _ResponseMeta]:
        """Like _do_json but without auth headers."""
        url = self.base_url + path
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"

        resp = self._client.request(
            method,
            url,
            headers=headers,
            content=json.dumps(body).encode() if body is not None else None,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    def _do_multipart(
        self,
        path: str,
        fields: dict[str, Any],
        files: dict[str, tuple[str, bytes, str]],
    ) -> tuple[Any, _ResponseMeta]:
        """POST multipart/form-data request."""
        url = self.base_url + path
        headers = self._headers()

        resp = self._client.post(
            url,
            headers=headers,
            data=fields,
            files=files,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    def _stream_sse(
        self,
        path: str,
        body: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        """POST and stream SSE events."""
        url = self.base_url + path
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        with httpx.Client() as stream_client:
            with stream_client.stream(
                "POST",
                url,
                headers=headers,
                content=json.dumps(body).encode(),
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    resp.read()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                yield from _iter_sse_lines(resp.iter_lines())

    # -- Chat ---------------------------------------------------------------

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat request.

        Can be called with keyword convenience args:
            client.chat(model="...", messages=[{"role": "user", "content": "Hi"}])

        Or with a full ChatRequest:
            client.chat(request=ChatRequest(model="...", messages=[...]))
        """
        req = self._build_chat_request(model, messages, request, **kwargs)
        req.stream = False
        data, meta = self._do_json("POST", "/qai/v1/chat", req.to_dict())
        resp = ChatResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if not resp.model:
            resp.model = meta.model
        return resp

    def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        **kwargs: Any,
    ) -> Iterator[StreamEvent]:
        """Send a streaming chat request. Yields StreamEvent objects."""
        req = self._build_chat_request(model, messages, request, **kwargs)
        req.stream = True

        url = self.base_url + "/qai/v1/chat"
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        # Use a separate client without timeout for streaming
        with httpx.Client() as stream_client:
            with stream_client.stream(
                "POST",
                url,
                headers=headers,
                content=json.dumps(req.to_dict()).encode(),
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    resp.read()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield StreamEvent(type="done", done=True)
                        return
                    try:
                        yield _parse_sse_event(payload)
                    except json.JSONDecodeError as exc:
                        yield StreamEvent(type="error", error=f"parse SSE: {exc}")
                        return

    @staticmethod
    def _build_chat_request(
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None,
        request: ChatRequest | None,
        **kwargs: Any,
    ) -> ChatRequest:
        if request is not None:
            return request
        from .types import ChatMessage
        msg_list: list[ChatMessage] = []
        for m in (messages or []):
            if isinstance(m, dict):
                msg_list.append(ChatMessage(
                    role=m["role"],
                    content=m.get("content", ""),
                    tool_call_id=m.get("tool_call_id"),
                    is_error=m.get("is_error", False),
                ))
            elif isinstance(m, ChatMessage):
                msg_list.append(m)
            else:
                msg_list.append(m)  # type: ignore[arg-type]
        return ChatRequest(model=model, messages=msg_list, **kwargs)

    # -- Session Chat -------------------------------------------------------

    def chat_session(
        self,
        message: str,
        *,
        model: str | None = None,
        session_id: str | None = None,
        tools: list[ChatTool] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
        context_config: ContextConfig | None = None,
    ) -> SessionChatResponse | Iterator[StreamEvent]:
        """Send a session-based chat request.

        Returns SessionChatResponse for non-streaming, or an Iterator of
        StreamEvent for streaming.
        """
        req = SessionChatRequest(
            message=message,
            model=model,
            session_id=session_id,
            tools=tools,
            tool_results=tool_results,
            stream=stream,
            system_prompt=system_prompt,
            context_config=context_config,
        )
        if stream:
            return self._stream_sse("/qai/v1/chat/session", req.to_dict())
        data, meta = self._do_json("POST", "/qai/v1/chat/session", req.to_dict())
        resp = SessionChatResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Agent --------------------------------------------------------------

    def agent_run(
        self,
        task: str,
        *,
        conductor_model: str | None = None,
        workers: list[AgentWorker] | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Run an agent task. Returns an SSE stream of events."""
        req = AgentRunRequest(
            task=task,
            conductor_model=conductor_model,
            workers=workers,
            max_steps=max_steps,
            system_prompt=system_prompt,
            session_id=session_id,
        )
        return self._stream_sse("/qai/v1/agent", req.to_dict())

    def mission_run(
        self,
        goal: str,
        *,
        strategy: str | None = None,
        conductor_model: str | None = None,
        workers: list[AgentWorker] | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> Iterator[StreamEvent]:
        """Run a mission. Returns an SSE stream of events."""
        req = MissionRunRequest(
            goal=goal,
            strategy=strategy,
            conductor_model=conductor_model,
            workers=workers,
            max_steps=max_steps,
            system_prompt=system_prompt,
            session_id=session_id,
        )
        return self._stream_sse("/qai/v1/missions", req.to_dict())

    # -- API Keys -----------------------------------------------------------

    def create_key(
        self,
        name: str,
        *,
        endpoints: list[str] | None = None,
        spend_cap_usd: float | None = None,
        rate_limit: int | None = None,
    ) -> APIKeyCreateResponse:
        """Create a new API key."""
        req = APIKeyCreateRequest(
            name=name,
            endpoints=endpoints,
            spend_cap_usd=spend_cap_usd,
            rate_limit=rate_limit,
        )
        data, _ = self._do_json("POST", "/qai/v1/keys", req.to_dict())
        return APIKeyCreateResponse.from_dict(data)

    def list_keys(self) -> APIKeyListResponse:
        """List all API keys for the account."""
        data, _ = self._do_json("GET", "/qai/v1/keys")
        return APIKeyListResponse.from_dict(data)

    def revoke_key(self, key_id: str) -> None:
        """Revoke an API key."""
        self._do_json("DELETE", f"/qai/v1/keys/{key_id}")

    # -- Compute ------------------------------------------------------------

    def compute_templates(self) -> list[ComputeTemplate]:
        """List available compute templates."""
        data, _ = self._do_json("GET", "/qai/v1/compute/templates")
        return [
            ComputeTemplate.from_dict(t)
            for t in data.get("templates", [])
        ]

    def compute_provision(
        self,
        template: str,
        *,
        zone: str | None = None,
        spot: bool = False,
        auto_teardown_minutes: int = 30,
        ssh_public_key: str | None = None,
    ) -> ComputeProvisionResponse:
        """Provision a new compute instance."""
        req = ComputeProvisionRequest(
            template=template,
            zone=zone,
            spot=spot,
            auto_teardown_minutes=auto_teardown_minutes,
            ssh_public_key=ssh_public_key,
        )
        data, _ = self._do_json("POST", "/qai/v1/compute/provision", req.to_dict())
        return ComputeProvisionResponse.from_dict(data)

    def compute_instances(self) -> list[ComputeInstance]:
        """List all compute instances."""
        data, _ = self._do_json("GET", "/qai/v1/compute/instances")
        return [
            ComputeInstance.from_dict(i)
            for i in data.get("instances", [])
        ]

    def compute_instance(self, instance_id: str) -> ComputeInstance:
        """Get a specific compute instance."""
        data, _ = self._do_json("GET", f"/qai/v1/compute/instance/{instance_id}")
        return ComputeInstance.from_dict(data)

    def compute_delete(self, instance_id: str) -> None:
        """Delete a compute instance."""
        self._do_json("DELETE", f"/qai/v1/compute/instance/{instance_id}")

    def compute_ssh_key(
        self,
        instance_id: str,
        public_key: str,
        username: str = "cosmic",
    ) -> None:
        """Add an SSH key to a compute instance."""
        body: dict[str, Any] = {"public_key": public_key, "username": username}
        self._do_json("POST", f"/qai/v1/compute/instance/{instance_id}/ssh-key", body)

    def compute_keepalive(self, instance_id: str) -> None:
        """Send a keepalive to extend a compute instance's auto-teardown timer."""
        self._do_json("POST", f"/qai/v1/compute/instance/{instance_id}/keepalive")

    # -- Voices -------------------------------------------------------------

    def list_voices(self) -> VoiceListResponse:
        """List all available voices."""
        data, _ = self._do_json("GET", "/qai/v1/voices")
        return VoiceListResponse.from_dict(data)

    def clone_voice(
        self,
        name: str,
        audio_file: bytes,
        *,
        filename: str = "voice.mp3",
        content_type: str = "audio/mpeg",
    ) -> VoiceCloneResponse:
        """Clone a voice from an audio sample (multipart upload)."""
        data, _ = self._do_multipart(
            "/qai/v1/voices/clone",
            fields={"name": name},
            files={"audio_file": (filename, audio_file, content_type)},
        )
        return VoiceCloneResponse.from_dict(data)

    def delete_voice(self, voice_id: str) -> None:
        """Delete a cloned voice."""
        self._do_json("DELETE", f"/qai/v1/voices/{voice_id}")

    # -- Image --------------------------------------------------------------

    def generate_image(self, request: ImageRequest) -> ImageResponse:
        """Generate images from a text prompt."""
        data, meta = self._do_json("POST", "/qai/v1/images/generate", request.to_dict())
        resp = ImageResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def edit_image(self, request: ImageEditRequest) -> ImageEditResponse:
        """Edit images using an AI model."""
        data, meta = self._do_json("POST", "/qai/v1/images/edit", request.to_dict())
        resp = ImageEditResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Video --------------------------------------------------------------

    def generate_video(self, request: VideoRequest) -> VideoResponse:
        """Generate a video from a text prompt."""
        data, meta = self._do_json("POST", "/qai/v1/video/generate", request.to_dict())
        resp = VideoResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def video_studio(
        self,
        avatar_id: str,
        script: str,
        voice_id: str,
    ) -> JobStatusResponse:
        """Create a HeyGen studio video."""
        req = VideoStudioRequest(avatar_id=avatar_id, script=script, voice_id=voice_id)
        data, meta = self._do_json("POST", "/qai/v1/video/studio", req.to_dict())
        return JobStatusResponse.from_dict(data)

    def video_translate(
        self,
        video_url: str,
        target_language: str,
    ) -> JobStatusResponse:
        """Translate a video to another language."""
        req = VideoTranslateRequest(video_url=video_url, target_language=target_language)
        data, meta = self._do_json("POST", "/qai/v1/video/translate", req.to_dict())
        return JobStatusResponse.from_dict(data)

    def video_photo_avatar(
        self,
        photo_url: str,
        script: str,
    ) -> JobStatusResponse:
        """Generate a video from a photo avatar."""
        req = VideoPhotoAvatarRequest(photo_url=photo_url, script=script)
        data, meta = self._do_json("POST", "/qai/v1/video/photo-avatar", req.to_dict())
        return JobStatusResponse.from_dict(data)

    def video_digital_twin(
        self,
        avatar_id: str,
        script: str,
    ) -> JobStatusResponse:
        """Generate a video with a digital twin."""
        req = VideoDigitalTwinRequest(avatar_id=avatar_id, script=script)
        data, meta = self._do_json("POST", "/qai/v1/video/digital-twin", req.to_dict())
        return JobStatusResponse.from_dict(data)

    def video_avatars(self) -> list[HeyGenAvatar]:
        """List available HeyGen avatars."""
        data, _ = self._do_json("GET", "/qai/v1/video/avatars")
        return [HeyGenAvatar.from_dict(a) for a in data.get("avatars", [])]

    def video_templates(self) -> list[HeyGenTemplate]:
        """List available HeyGen video templates."""
        data, _ = self._do_json("GET", "/qai/v1/video/templates")
        return [HeyGenTemplate.from_dict(t) for t in data.get("templates", [])]

    def video_heygen_voices(self) -> list[HeyGenVoice]:
        """List available HeyGen voices."""
        data, _ = self._do_json("GET", "/qai/v1/video/heygen-voices")
        return [HeyGenVoice.from_dict(v) for v in data.get("voices", [])]

    # -- Audio --------------------------------------------------------------

    def speak(self, request: TTSRequest) -> TTSResponse:
        """Generate speech from text."""
        data, meta = self._do_json("POST", "/qai/v1/audio/tts", request.to_dict())
        resp = TTSResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def transcribe(self, request: STTRequest) -> STTResponse:
        """Convert speech to text."""
        data, meta = self._do_json("POST", "/qai/v1/audio/stt", request.to_dict())
        resp = STTResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def generate_music(self, request: MusicRequest) -> MusicResponse:
        """Generate music from a text prompt."""
        data, meta = self._do_json("POST", "/qai/v1/audio/music", request.to_dict())
        resp = MusicResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def sound_effects(
        self,
        prompt: str,
        duration_seconds: float | None = None,
    ) -> SoundEffectResponse:
        """Generate sound effects from a text prompt."""
        body: dict[str, Any] = {"prompt": prompt}
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        data, meta = self._do_json("POST", "/qai/v1/audio/sound-effects", body)
        resp = SoundEffectResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def dialogue(
        self,
        text: str,
        voices: list[DialogueVoice],
        *,
        model: str | None = None,
    ) -> AudioResponse:
        """Generate multi-voice dialogue audio."""
        req = DialogueRequest(text=text, voices=voices, model=model)
        data, meta = self._do_json("POST", "/qai/v1/audio/dialogue", req.to_dict())
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def speech_to_speech(
        self,
        audio: str,
        voice_id: str,
    ) -> AudioResponse:
        """Convert speech audio to a different voice."""
        body: dict[str, Any] = {"audio": audio, "voice_id": voice_id}
        data, meta = self._do_json("POST", "/qai/v1/audio/speech-to-speech", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def isolate_voice(self, audio: str) -> AudioResponse:
        """Isolate voice from background noise in an audio clip."""
        body: dict[str, Any] = {"audio": audio}
        data, meta = self._do_json("POST", "/qai/v1/audio/isolate", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def remix_voice(self, audio: str, voice_id: str) -> AudioResponse:
        """Remix audio with a different voice."""
        body: dict[str, Any] = {"audio": audio, "voice_id": voice_id}
        data, meta = self._do_json("POST", "/qai/v1/audio/remix", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def dub(
        self,
        audio: str,
        target_lang: str,
        *,
        source_lang: str | None = None,
    ) -> AudioResponse:
        """Dub audio into a target language."""
        body: dict[str, Any] = {"audio": audio, "target_lang": target_lang}
        if source_lang is not None:
            body["source_lang"] = source_lang
        data, meta = self._do_json("POST", "/qai/v1/audio/dub", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def align(self, audio: str, text: str) -> AlignmentResponse:
        """Align audio with text to get word-level timing."""
        body: dict[str, Any] = {"audio": audio, "text": text}
        data, meta = self._do_json("POST", "/qai/v1/audio/align", body)
        resp = AlignmentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def voice_design(self, description: str) -> VoiceDesignResponse:
        """Design a new voice from a text description."""
        body: dict[str, Any] = {"description": description}
        data, meta = self._do_json("POST", "/qai/v1/audio/voice-design", body)
        resp = VoiceDesignResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def starfish_tts(self, text: str, voice_id: str) -> AudioResponse:
        """Generate speech using Starfish TTS engine."""
        body: dict[str, Any] = {"text": text, "voice_id": voice_id}
        data, meta = self._do_json("POST", "/qai/v1/audio/starfish-tts", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Embeddings ---------------------------------------------------------

    def embed(self, request: EmbedRequest) -> EmbedResponse:
        """Generate text embeddings."""
        data, meta = self._do_json("POST", "/qai/v1/embeddings", request.to_dict())
        resp = EmbedResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Documents ----------------------------------------------------------

    def extract_document(self, request: DocumentRequest) -> DocumentResponse:
        """Extract text content from a document (PDF, image, etc.)."""
        data, meta = self._do_json("POST", "/qai/v1/documents/extract", request.to_dict())
        resp = DocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def chunk_document(
        self,
        text: str,
        *,
        chunk_size: int | None = None,
    ) -> ChunkDocumentResponse:
        """Chunk a document into smaller pieces."""
        req = ChunkDocumentRequest(text=text, chunk_size=chunk_size)
        data, meta = self._do_json("POST", "/qai/v1/documents/chunk", req.to_dict())
        resp = ChunkDocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def process_document(self, text: str) -> ProcessDocumentResponse:
        """Process a document."""
        req = ProcessDocumentRequest(text=text)
        data, meta = self._do_json("POST", "/qai/v1/documents/process", req.to_dict())
        resp = ProcessDocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- RAG ----------------------------------------------------------------

    def rag_search(self, request: RAGSearchRequest) -> RAGSearchResponse:
        """Search Vertex AI RAG corpora."""
        data, meta = self._do_json("POST", "/qai/v1/rag/search", request.to_dict())
        resp = RAGSearchResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def rag_corpora(self) -> list[RAGCorpus]:
        """List available Vertex AI RAG corpora."""
        data, _ = self._do_json("GET", "/qai/v1/rag/corpora")
        return [
            RAGCorpus(
                name=c.get("name", ""),
                display_name=c.get("displayName", ""),
                description=c.get("description", ""),
                state=c.get("state", ""),
            )
            for c in data.get("corpora", [])
        ]

    def surreal_rag_search(self, request: SurrealRAGSearchRequest) -> SurrealRAGSearchResponse:
        """Search provider API docs via SurrealDB vector search."""
        data, meta = self._do_json("POST", "/qai/v1/rag/surreal/search", request.to_dict())
        resp = SurrealRAGSearchResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    def surreal_rag_providers(self) -> SurrealRAGProvidersResponse:
        """List available SurrealDB RAG providers."""
        data, _ = self._do_json("GET", "/qai/v1/rag/surreal/providers")
        return SurrealRAGProvidersResponse.from_dict(data)

    # -- Models -------------------------------------------------------------

    def list_models(self) -> list[ModelInfo]:
        """Return all available models."""
        data, _ = self._do_json("GET", "/qai/v1/models")
        return [
            ModelInfo(
                id=m.get("id", ""),
                provider=m.get("provider", ""),
                display_name=m.get("display_name", ""),
                input_per_million=m.get("input_per_million", 0.0),
                output_per_million=m.get("output_per_million", 0.0),
            )
            for m in data.get("models", [])
        ]

    def get_pricing(self) -> list[PricingInfo]:
        """Return the complete pricing table."""
        data, _ = self._do_json("GET", "/qai/v1/pricing")
        return [
            PricingInfo(
                id=p.get("id", ""),
                provider=p.get("provider", ""),
                display_name=p.get("display_name", ""),
                input_per_million=p.get("input_per_million", 0.0),
                output_per_million=p.get("output_per_million", 0.0),
            )
            for p in data.get("pricing", [])
        ]

    # -- Account ------------------------------------------------------------

    def account_balance(self) -> BalanceResponse:
        """Get the account credit balance."""
        data, _ = self._do_json("GET", "/qai/v1/account/balance")
        return BalanceResponse.from_dict(data)

    def account_usage(
        self,
        limit: int | None = None,
        start_after: str | None = None,
    ) -> UsageResponse:
        """Get paginated usage history."""
        params: list[str] = []
        if limit is not None:
            params.append(f"limit={limit}")
        if start_after is not None:
            params.append(f"start_after={start_after}")
        path = "/qai/v1/account/usage"
        if params:
            path += "?" + "&".join(params)
        data, _ = self._do_json("GET", path)
        return UsageResponse.from_dict(data)

    def account_usage_summary(self, months: int | None = None) -> UsageSummaryResponse:
        """Get monthly usage summary."""
        path = "/qai/v1/account/usage/summary"
        if months is not None:
            path += f"?months={months}"
        data, _ = self._do_json("GET", path)
        return UsageSummaryResponse.from_dict(data)

    def account_pricing(self) -> PricingResponse:
        """Get the full pricing table (model to pricing entry map)."""
        data, _ = self._do_json("GET", "/qai/v1/pricing")
        return PricingResponse.from_dict(data)

    # -- Jobs ---------------------------------------------------------------

    def create_job(
        self,
        job_type: str,
        params: dict[str, Any],
    ) -> JobCreateResponse:
        """Create an async job. Returns the job ID for polling."""
        body: dict[str, Any] = {"type": job_type, "params": params}
        data, _ = self._do_json("POST", "/qai/v1/jobs", body)
        return JobCreateResponse.from_dict(data)

    def get_job(self, job_id: str) -> JobStatusResponse:
        """Check the status of an async job."""
        data, _ = self._do_json("GET", f"/qai/v1/jobs/{job_id}")
        return JobStatusResponse.from_dict(data)

    def list_jobs(self) -> JobListResponse:
        """List all jobs for the account."""
        data, _ = self._do_json("GET", "/qai/v1/jobs")
        return JobListResponse.from_dict(data)

    def poll_job(
        self,
        job_id: str,
        interval: float = 5.0,
        max_attempts: int = 120,
    ) -> JobStatusResponse:
        """Poll a job until completion or timeout."""
        for _ in range(max_attempts):
            time.sleep(interval)
            status = self.get_job(job_id)
            if status.status in ("completed", "failed"):
                return status
        return JobStatusResponse(
            job_id=job_id,
            status="timeout",
            error=f"Job polling timed out after {max_attempts} attempts",
        )

    # -- Contact ------------------------------------------------------------

    def contact(
        self,
        name: str,
        email: str,
        message: str,
    ) -> ContactResponse:
        """Submit a contact form (no auth required)."""
        req = ContactRequest(name=name, email=email, message=message)
        data, _ = self._do_json_no_auth("POST", "/qai/v1/contact", req.to_dict())
        return ContactResponse.from_dict(data)


# ---------------------------------------------------------------------------
# Async client
# ---------------------------------------------------------------------------


class AsyncClient:
    """Asynchronous Quantum AI API client."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = DEFAULT_TIMEOUT,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self._client = http_client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = http_client is None

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _do_json(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[Any, _ResponseMeta]:
        url = self.base_url + path
        headers = self._headers()
        if body is not None:
            headers["Content-Type"] = "application/json"

        resp = await self._client.request(
            method,
            url,
            headers=headers,
            content=json.dumps(body).encode() if body is not None else None,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    async def _do_json_no_auth(
        self,
        method: str,
        path: str,
        body: dict[str, Any] | None = None,
    ) -> tuple[Any, _ResponseMeta]:
        """Like _do_json but without auth headers."""
        url = self.base_url + path
        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"

        resp = await self._client.request(
            method,
            url,
            headers=headers,
            content=json.dumps(body).encode() if body is not None else None,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    async def _do_multipart(
        self,
        path: str,
        fields: dict[str, Any],
        files: dict[str, tuple[str, bytes, str]],
    ) -> tuple[Any, _ResponseMeta]:
        """POST multipart/form-data request."""
        url = self.base_url + path
        headers = self._headers()

        resp = await self._client.post(
            url,
            headers=headers,
            data=fields,
            files=files,
        )
        meta = _ResponseMeta(resp.headers)

        if resp.status_code < 200 or resp.status_code >= 300:
            raise _parse_api_error(resp, meta.request_id)

        data = resp.json() if resp.content else None
        return data, meta

    async def _stream_sse(
        self,
        path: str,
        body: dict[str, Any],
    ) -> AsyncIterator[StreamEvent]:
        """POST and stream SSE events."""
        url = self.base_url + path
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        stream_client = httpx.AsyncClient()
        try:
            async with stream_client.stream(
                "POST",
                url,
                headers=headers,
                content=json.dumps(body).encode(),
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    await resp.aread()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield StreamEvent(type="done", done=True)
                        return
                    try:
                        yield _parse_sse_event(payload)
                    except json.JSONDecodeError as exc:
                        yield StreamEvent(type="error", error=f"parse SSE: {exc}")
                        return
        finally:
            await stream_client.aclose()

    # -- Chat ---------------------------------------------------------------

    async def chat(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        **kwargs: Any,
    ) -> ChatResponse:
        """Send a non-streaming chat request."""
        req = Client._build_chat_request(model, messages, request, **kwargs)
        req.stream = False
        data, meta = await self._do_json("POST", "/qai/v1/chat", req.to_dict())
        resp = ChatResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        if not resp.model:
            resp.model = meta.model
        return resp

    async def chat_stream(
        self,
        model: str,
        messages: list[dict[str, Any] | ChatRequest] | None = None,
        *,
        request: ChatRequest | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamEvent]:
        """Send a streaming chat request. Yields StreamEvent objects."""
        req = Client._build_chat_request(model, messages, request, **kwargs)
        req.stream = True

        url = self.base_url + "/qai/v1/chat"
        headers = self._headers()
        headers["Content-Type"] = "application/json"
        headers["Accept"] = "text/event-stream"

        stream_client = httpx.AsyncClient()
        try:
            async with stream_client.stream(
                "POST",
                url,
                headers=headers,
                content=json.dumps(req.to_dict()).encode(),
            ) as resp:
                if resp.status_code < 200 or resp.status_code >= 300:
                    await resp.aread()
                    meta = _ResponseMeta(resp.headers)
                    raise _parse_api_error(resp, meta.request_id)

                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    payload = line[6:]
                    if payload == "[DONE]":
                        yield StreamEvent(type="done", done=True)
                        return
                    try:
                        yield _parse_sse_event(payload)
                    except json.JSONDecodeError as exc:
                        yield StreamEvent(type="error", error=f"parse SSE: {exc}")
                        return
        finally:
            await stream_client.aclose()

    # -- Session Chat -------------------------------------------------------

    async def chat_session(
        self,
        message: str,
        *,
        model: str | None = None,
        session_id: str | None = None,
        tools: list[ChatTool] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        stream: bool = False,
        system_prompt: str | None = None,
        context_config: ContextConfig | None = None,
    ) -> SessionChatResponse | AsyncIterator[StreamEvent]:
        """Send a session-based chat request.

        Returns SessionChatResponse for non-streaming, or an AsyncIterator of
        StreamEvent for streaming.
        """
        req = SessionChatRequest(
            message=message,
            model=model,
            session_id=session_id,
            tools=tools,
            tool_results=tool_results,
            stream=stream,
            system_prompt=system_prompt,
            context_config=context_config,
        )
        if stream:
            return self._stream_sse("/qai/v1/chat/session", req.to_dict())
        data, meta = await self._do_json("POST", "/qai/v1/chat/session", req.to_dict())
        resp = SessionChatResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Agent --------------------------------------------------------------

    async def agent_run(
        self,
        task: str,
        *,
        conductor_model: str | None = None,
        workers: list[AgentWorker] | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run an agent task. Returns an SSE stream of events."""
        req = AgentRunRequest(
            task=task,
            conductor_model=conductor_model,
            workers=workers,
            max_steps=max_steps,
            system_prompt=system_prompt,
            session_id=session_id,
        )
        return self._stream_sse("/qai/v1/agent", req.to_dict())

    async def mission_run(
        self,
        goal: str,
        *,
        strategy: str | None = None,
        conductor_model: str | None = None,
        workers: list[AgentWorker] | None = None,
        max_steps: int | None = None,
        system_prompt: str | None = None,
        session_id: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Run a mission. Returns an SSE stream of events."""
        req = MissionRunRequest(
            goal=goal,
            strategy=strategy,
            conductor_model=conductor_model,
            workers=workers,
            max_steps=max_steps,
            system_prompt=system_prompt,
            session_id=session_id,
        )
        return self._stream_sse("/qai/v1/missions", req.to_dict())

    # -- API Keys -----------------------------------------------------------

    async def create_key(
        self,
        name: str,
        *,
        endpoints: list[str] | None = None,
        spend_cap_usd: float | None = None,
        rate_limit: int | None = None,
    ) -> APIKeyCreateResponse:
        """Create a new API key."""
        req = APIKeyCreateRequest(
            name=name,
            endpoints=endpoints,
            spend_cap_usd=spend_cap_usd,
            rate_limit=rate_limit,
        )
        data, _ = await self._do_json("POST", "/qai/v1/keys", req.to_dict())
        return APIKeyCreateResponse.from_dict(data)

    async def list_keys(self) -> APIKeyListResponse:
        """List all API keys for the account."""
        data, _ = await self._do_json("GET", "/qai/v1/keys")
        return APIKeyListResponse.from_dict(data)

    async def revoke_key(self, key_id: str) -> None:
        """Revoke an API key."""
        await self._do_json("DELETE", f"/qai/v1/keys/{key_id}")

    # -- Compute ------------------------------------------------------------

    async def compute_templates(self) -> list[ComputeTemplate]:
        """List available compute templates."""
        data, _ = await self._do_json("GET", "/qai/v1/compute/templates")
        return [
            ComputeTemplate.from_dict(t)
            for t in data.get("templates", [])
        ]

    async def compute_provision(
        self,
        template: str,
        *,
        zone: str | None = None,
        spot: bool = False,
        auto_teardown_minutes: int = 30,
        ssh_public_key: str | None = None,
    ) -> ComputeProvisionResponse:
        """Provision a new compute instance."""
        req = ComputeProvisionRequest(
            template=template,
            zone=zone,
            spot=spot,
            auto_teardown_minutes=auto_teardown_minutes,
            ssh_public_key=ssh_public_key,
        )
        data, _ = await self._do_json("POST", "/qai/v1/compute/provision", req.to_dict())
        return ComputeProvisionResponse.from_dict(data)

    async def compute_instances(self) -> list[ComputeInstance]:
        """List all compute instances."""
        data, _ = await self._do_json("GET", "/qai/v1/compute/instances")
        return [
            ComputeInstance.from_dict(i)
            for i in data.get("instances", [])
        ]

    async def compute_instance(self, instance_id: str) -> ComputeInstance:
        """Get a specific compute instance."""
        data, _ = await self._do_json("GET", f"/qai/v1/compute/instance/{instance_id}")
        return ComputeInstance.from_dict(data)

    async def compute_delete(self, instance_id: str) -> None:
        """Delete a compute instance."""
        await self._do_json("DELETE", f"/qai/v1/compute/instance/{instance_id}")

    async def compute_ssh_key(
        self,
        instance_id: str,
        public_key: str,
        username: str = "cosmic",
    ) -> None:
        """Add an SSH key to a compute instance."""
        body: dict[str, Any] = {"public_key": public_key, "username": username}
        await self._do_json("POST", f"/qai/v1/compute/instance/{instance_id}/ssh-key", body)

    async def compute_keepalive(self, instance_id: str) -> None:
        """Send a keepalive to extend a compute instance's auto-teardown timer."""
        await self._do_json("POST", f"/qai/v1/compute/instance/{instance_id}/keepalive")

    # -- Voices -------------------------------------------------------------

    async def list_voices(self) -> VoiceListResponse:
        """List all available voices."""
        data, _ = await self._do_json("GET", "/qai/v1/voices")
        return VoiceListResponse.from_dict(data)

    async def clone_voice(
        self,
        name: str,
        audio_file: bytes,
        *,
        filename: str = "voice.mp3",
        content_type: str = "audio/mpeg",
    ) -> VoiceCloneResponse:
        """Clone a voice from an audio sample (multipart upload)."""
        data, _ = await self._do_multipart(
            "/qai/v1/voices/clone",
            fields={"name": name},
            files={"audio_file": (filename, audio_file, content_type)},
        )
        return VoiceCloneResponse.from_dict(data)

    async def delete_voice(self, voice_id: str) -> None:
        """Delete a cloned voice."""
        await self._do_json("DELETE", f"/qai/v1/voices/{voice_id}")

    # -- Image --------------------------------------------------------------

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        data, meta = await self._do_json("POST", "/qai/v1/images/generate", request.to_dict())
        resp = ImageResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def edit_image(self, request: ImageEditRequest) -> ImageEditResponse:
        data, meta = await self._do_json("POST", "/qai/v1/images/edit", request.to_dict())
        resp = ImageEditResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Video --------------------------------------------------------------

    async def generate_video(self, request: VideoRequest) -> VideoResponse:
        data, meta = await self._do_json("POST", "/qai/v1/video/generate", request.to_dict())
        resp = VideoResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def video_studio(
        self,
        avatar_id: str,
        script: str,
        voice_id: str,
    ) -> JobStatusResponse:
        """Create a HeyGen studio video."""
        req = VideoStudioRequest(avatar_id=avatar_id, script=script, voice_id=voice_id)
        data, meta = await self._do_json("POST", "/qai/v1/video/studio", req.to_dict())
        return JobStatusResponse.from_dict(data)

    async def video_translate(
        self,
        video_url: str,
        target_language: str,
    ) -> JobStatusResponse:
        """Translate a video to another language."""
        req = VideoTranslateRequest(video_url=video_url, target_language=target_language)
        data, meta = await self._do_json("POST", "/qai/v1/video/translate", req.to_dict())
        return JobStatusResponse.from_dict(data)

    async def video_photo_avatar(
        self,
        photo_url: str,
        script: str,
    ) -> JobStatusResponse:
        """Generate a video from a photo avatar."""
        req = VideoPhotoAvatarRequest(photo_url=photo_url, script=script)
        data, meta = await self._do_json("POST", "/qai/v1/video/photo-avatar", req.to_dict())
        return JobStatusResponse.from_dict(data)

    async def video_digital_twin(
        self,
        avatar_id: str,
        script: str,
    ) -> JobStatusResponse:
        """Generate a video with a digital twin."""
        req = VideoDigitalTwinRequest(avatar_id=avatar_id, script=script)
        data, meta = await self._do_json("POST", "/qai/v1/video/digital-twin", req.to_dict())
        return JobStatusResponse.from_dict(data)

    async def video_avatars(self) -> list[HeyGenAvatar]:
        """List available HeyGen avatars."""
        data, _ = await self._do_json("GET", "/qai/v1/video/avatars")
        return [HeyGenAvatar.from_dict(a) for a in data.get("avatars", [])]

    async def video_templates(self) -> list[HeyGenTemplate]:
        """List available HeyGen video templates."""
        data, _ = await self._do_json("GET", "/qai/v1/video/templates")
        return [HeyGenTemplate.from_dict(t) for t in data.get("templates", [])]

    async def video_heygen_voices(self) -> list[HeyGenVoice]:
        """List available HeyGen voices."""
        data, _ = await self._do_json("GET", "/qai/v1/video/heygen-voices")
        return [HeyGenVoice.from_dict(v) for v in data.get("voices", [])]

    # -- Audio --------------------------------------------------------------

    async def speak(self, request: TTSRequest) -> TTSResponse:
        data, meta = await self._do_json("POST", "/qai/v1/audio/tts", request.to_dict())
        resp = TTSResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def transcribe(self, request: STTRequest) -> STTResponse:
        data, meta = await self._do_json("POST", "/qai/v1/audio/stt", request.to_dict())
        resp = STTResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def generate_music(self, request: MusicRequest) -> MusicResponse:
        data, meta = await self._do_json("POST", "/qai/v1/audio/music", request.to_dict())
        resp = MusicResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def sound_effects(
        self,
        prompt: str,
        duration_seconds: float | None = None,
    ) -> SoundEffectResponse:
        """Generate sound effects from a text prompt."""
        body: dict[str, Any] = {"prompt": prompt}
        if duration_seconds is not None:
            body["duration_seconds"] = duration_seconds
        data, meta = await self._do_json("POST", "/qai/v1/audio/sound-effects", body)
        resp = SoundEffectResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def dialogue(
        self,
        text: str,
        voices: list[DialogueVoice],
        *,
        model: str | None = None,
    ) -> AudioResponse:
        """Generate multi-voice dialogue audio."""
        req = DialogueRequest(text=text, voices=voices, model=model)
        data, meta = await self._do_json("POST", "/qai/v1/audio/dialogue", req.to_dict())
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def speech_to_speech(
        self,
        audio: str,
        voice_id: str,
    ) -> AudioResponse:
        """Convert speech audio to a different voice."""
        body: dict[str, Any] = {"audio": audio, "voice_id": voice_id}
        data, meta = await self._do_json("POST", "/qai/v1/audio/speech-to-speech", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def isolate_voice(self, audio: str) -> AudioResponse:
        """Isolate voice from background noise in an audio clip."""
        body: dict[str, Any] = {"audio": audio}
        data, meta = await self._do_json("POST", "/qai/v1/audio/isolate", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def remix_voice(self, audio: str, voice_id: str) -> AudioResponse:
        """Remix audio with a different voice."""
        body: dict[str, Any] = {"audio": audio, "voice_id": voice_id}
        data, meta = await self._do_json("POST", "/qai/v1/audio/remix", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def dub(
        self,
        audio: str,
        target_lang: str,
        *,
        source_lang: str | None = None,
    ) -> AudioResponse:
        """Dub audio into a target language."""
        body: dict[str, Any] = {"audio": audio, "target_lang": target_lang}
        if source_lang is not None:
            body["source_lang"] = source_lang
        data, meta = await self._do_json("POST", "/qai/v1/audio/dub", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def align(self, audio: str, text: str) -> AlignmentResponse:
        """Align audio with text to get word-level timing."""
        body: dict[str, Any] = {"audio": audio, "text": text}
        data, meta = await self._do_json("POST", "/qai/v1/audio/align", body)
        resp = AlignmentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def voice_design(self, description: str) -> VoiceDesignResponse:
        """Design a new voice from a text description."""
        body: dict[str, Any] = {"description": description}
        data, meta = await self._do_json("POST", "/qai/v1/audio/voice-design", body)
        resp = VoiceDesignResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def starfish_tts(self, text: str, voice_id: str) -> AudioResponse:
        """Generate speech using Starfish TTS engine."""
        body: dict[str, Any] = {"text": text, "voice_id": voice_id}
        data, meta = await self._do_json("POST", "/qai/v1/audio/starfish-tts", body)
        resp = AudioResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Embeddings ---------------------------------------------------------

    async def embed(self, request: EmbedRequest) -> EmbedResponse:
        data, meta = await self._do_json("POST", "/qai/v1/embeddings", request.to_dict())
        resp = EmbedResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- Documents ----------------------------------------------------------

    async def extract_document(self, request: DocumentRequest) -> DocumentResponse:
        data, meta = await self._do_json("POST", "/qai/v1/documents/extract", request.to_dict())
        resp = DocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def chunk_document(
        self,
        text: str,
        *,
        chunk_size: int | None = None,
    ) -> ChunkDocumentResponse:
        """Chunk a document into smaller pieces."""
        req = ChunkDocumentRequest(text=text, chunk_size=chunk_size)
        data, meta = await self._do_json("POST", "/qai/v1/documents/chunk", req.to_dict())
        resp = ChunkDocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def process_document(self, text: str) -> ProcessDocumentResponse:
        """Process a document."""
        req = ProcessDocumentRequest(text=text)
        data, meta = await self._do_json("POST", "/qai/v1/documents/process", req.to_dict())
        resp = ProcessDocumentResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    # -- RAG ----------------------------------------------------------------

    async def rag_search(self, request: RAGSearchRequest) -> RAGSearchResponse:
        data, meta = await self._do_json("POST", "/qai/v1/rag/search", request.to_dict())
        resp = RAGSearchResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def rag_corpora(self) -> list[RAGCorpus]:
        data, _ = await self._do_json("GET", "/qai/v1/rag/corpora")
        return [
            RAGCorpus(
                name=c.get("name", ""),
                display_name=c.get("displayName", ""),
                description=c.get("description", ""),
                state=c.get("state", ""),
            )
            for c in data.get("corpora", [])
        ]

    async def surreal_rag_search(self, request: SurrealRAGSearchRequest) -> SurrealRAGSearchResponse:
        data, meta = await self._do_json("POST", "/qai/v1/rag/surreal/search", request.to_dict())
        resp = SurrealRAGSearchResponse.from_dict(data)
        if not resp.cost_ticks:
            resp.cost_ticks = meta.cost_ticks
        if not resp.request_id:
            resp.request_id = meta.request_id
        return resp

    async def surreal_rag_providers(self) -> SurrealRAGProvidersResponse:
        """List available SurrealDB RAG providers."""
        data, _ = await self._do_json("GET", "/qai/v1/rag/surreal/providers")
        return SurrealRAGProvidersResponse.from_dict(data)

    # -- Models -------------------------------------------------------------

    async def list_models(self) -> list[ModelInfo]:
        data, _ = await self._do_json("GET", "/qai/v1/models")
        return [
            ModelInfo(
                id=m.get("id", ""),
                provider=m.get("provider", ""),
                display_name=m.get("display_name", ""),
                input_per_million=m.get("input_per_million", 0.0),
                output_per_million=m.get("output_per_million", 0.0),
            )
            for m in data.get("models", [])
        ]

    async def get_pricing(self) -> list[PricingInfo]:
        data, _ = await self._do_json("GET", "/qai/v1/pricing")
        return [
            PricingInfo(
                id=p.get("id", ""),
                provider=p.get("provider", ""),
                display_name=p.get("display_name", ""),
                input_per_million=p.get("input_per_million", 0.0),
                output_per_million=p.get("output_per_million", 0.0),
            )
            for p in data.get("pricing", [])
        ]

    # -- Account ------------------------------------------------------------

    async def account_balance(self) -> BalanceResponse:
        """Get the account credit balance."""
        data, _ = await self._do_json("GET", "/qai/v1/account/balance")
        return BalanceResponse.from_dict(data)

    async def account_usage(
        self,
        limit: int | None = None,
        start_after: str | None = None,
    ) -> UsageResponse:
        """Get paginated usage history."""
        params: list[str] = []
        if limit is not None:
            params.append(f"limit={limit}")
        if start_after is not None:
            params.append(f"start_after={start_after}")
        path = "/qai/v1/account/usage"
        if params:
            path += "?" + "&".join(params)
        data, _ = await self._do_json("GET", path)
        return UsageResponse.from_dict(data)

    async def account_usage_summary(self, months: int | None = None) -> UsageSummaryResponse:
        """Get monthly usage summary."""
        path = "/qai/v1/account/usage/summary"
        if months is not None:
            path += f"?months={months}"
        data, _ = await self._do_json("GET", path)
        return UsageSummaryResponse.from_dict(data)

    async def account_pricing(self) -> PricingResponse:
        """Get the full pricing table (model to pricing entry map)."""
        data, _ = await self._do_json("GET", "/qai/v1/pricing")
        return PricingResponse.from_dict(data)

    # -- Jobs ---------------------------------------------------------------

    async def create_job(
        self,
        job_type: str,
        params: dict[str, Any],
    ) -> JobCreateResponse:
        """Create an async job. Returns the job ID for polling."""
        body: dict[str, Any] = {"type": job_type, "params": params}
        data, _ = await self._do_json("POST", "/qai/v1/jobs", body)
        return JobCreateResponse.from_dict(data)

    async def get_job(self, job_id: str) -> JobStatusResponse:
        """Check the status of an async job."""
        data, _ = await self._do_json("GET", f"/qai/v1/jobs/{job_id}")
        return JobStatusResponse.from_dict(data)

    async def list_jobs(self) -> JobListResponse:
        """List all jobs for the account."""
        data, _ = await self._do_json("GET", "/qai/v1/jobs")
        return JobListResponse.from_dict(data)

    async def poll_job(
        self,
        job_id: str,
        interval: float = 5.0,
        max_attempts: int = 120,
    ) -> JobStatusResponse:
        """Poll a job until completion or timeout."""
        import asyncio
        for _ in range(max_attempts):
            await asyncio.sleep(interval)
            status = await self.get_job(job_id)
            if status.status in ("completed", "failed"):
                return status
        return JobStatusResponse(
            job_id=job_id,
            status="timeout",
            error=f"Job polling timed out after {max_attempts} attempts",
        )

    # -- Contact ------------------------------------------------------------

    async def contact(
        self,
        name: str,
        email: str,
        message: str,
    ) -> ContactResponse:
        """Submit a contact form (no auth required)."""
        req = ContactRequest(name=name, email=email, message=message)
        data, _ = await self._do_json_no_auth("POST", "/qai/v1/contact", req.to_dict())
        return ContactResponse.from_dict(data)
