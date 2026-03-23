"""Extensions for Client and AsyncClient — adds missing parity methods.

Automatically patches methods onto Client and AsyncClient when imported.
Import order: __init__.py imports client_ext after client, so all methods
are available on the Client/AsyncClient classes.
"""

from __future__ import annotations

import json
import time
from typing import Any, Iterator, AsyncIterator

import httpx

from .client import Client, AsyncClient, _ResponseMeta, _parse_api_error
from .types import ChatRequest, JobCreateResponse, JobStatusResponse
from .types_ext import (
    SharedVoicesResponse,
    AddVoiceFromLibraryResponse,
    MusicFinetuneInfo,
    MusicFinetuneListResponse,
    BillingResponse,
    JobStreamEvent,
)


# ---------------------------------------------------------------------------
# Sync Client methods
# ---------------------------------------------------------------------------

def _sync_chat_job(
    self: Client,
    request: ChatRequest,
) -> JobCreateResponse:
    """Submit a chat completion as an async job.

    Useful for long-running models (e.g. Opus) where synchronous
    /qai/v1/chat may time out. Use stream_job() or poll_job() to get the result.
    """
    params = request.to_dict()
    params.pop("stream", None)
    return self.create_job("chat", params)


def _sync_stream_job(
    self: Client,
    job_id: str,
) -> Iterator[JobStreamEvent]:
    """Stream job progress via SSE. Yields JobStreamEvent objects.

    Events: "progress" (status update), "complete" (with result), "error".
    """
    url = self.base_url + f"/qai/v1/jobs/{job_id}/stream"
    headers = self._headers()
    headers["Accept"] = "text/event-stream"

    with httpx.Client() as stream_client:
        with stream_client.stream(
            "GET",
            url,
            headers=headers,
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
                    return
                try:
                    data = json.loads(payload)
                    event = JobStreamEvent.from_dict(data)
                except json.JSONDecodeError as exc:
                    yield JobStreamEvent(type="error", error=f"parse SSE: {exc}")
                    return

                yield event

                if event.type in ("complete", "error"):
                    return


def _sync_voice_library(
    self: Client,
    *,
    query: str | None = None,
    page_size: int | None = None,
    cursor: str | None = None,
    gender: str | None = None,
    language: str | None = None,
    use_case: str | None = None,
) -> SharedVoicesResponse:
    """Browse the shared voice library with optional filters."""
    params: list[str] = []
    if query is not None:
        params.append(f"query={query}")
    if page_size is not None:
        params.append(f"page_size={page_size}")
    if cursor is not None:
        params.append(f"cursor={cursor}")
    if gender is not None:
        params.append(f"gender={gender}")
    if language is not None:
        params.append(f"language={language}")
    if use_case is not None:
        params.append(f"use_case={use_case}")
    path = "/qai/v1/voices/library"
    if params:
        path += "?" + "&".join(params)
    data, _ = self._do_json("GET", path)
    return SharedVoicesResponse.from_dict(data)


def _sync_add_voice_from_library(
    self: Client,
    public_owner_id: str,
    voice_id: str,
    *,
    name: str | None = None,
) -> AddVoiceFromLibraryResponse:
    """Add a shared voice from the library to the user's account."""
    body: dict[str, Any] = {
        "public_owner_id": public_owner_id,
        "voice_id": voice_id,
    }
    if name is not None:
        body["name"] = name
    data, _ = self._do_json("POST", "/qai/v1/voices/library/add", body)
    return AddVoiceFromLibraryResponse.from_dict(data)


def _sync_create_finetune(
    self: Client,
    name: str,
    samples: list[str],
    *,
    description: str | None = None,
) -> MusicFinetuneInfo:
    """Create a new music finetune from audio samples (base64-encoded)."""
    body: dict[str, Any] = {"name": name, "samples": samples}
    if description is not None:
        body["description"] = description
    data, _ = self._do_json("POST", "/qai/v1/audio/finetunes", body)
    return MusicFinetuneInfo.from_dict(data)


def _sync_list_finetunes(self: Client) -> MusicFinetuneListResponse:
    """List all music finetunes for the authenticated user."""
    data, _ = self._do_json("GET", "/qai/v1/audio/finetunes")
    return MusicFinetuneListResponse.from_dict(data)


def _sync_delete_finetune(self: Client, finetune_id: str) -> None:
    """Delete a music finetune by ID."""
    self._do_json("DELETE", f"/qai/v1/audio/finetunes/{finetune_id}")


def _sync_compute_billing(
    self: Client,
    *,
    instance_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BillingResponse:
    """Query compute billing from BigQuery via the QAI backend."""
    body: dict[str, Any] = {}
    if instance_id is not None:
        body["instance_id"] = instance_id
    if start_date is not None:
        body["start_date"] = start_date
    if end_date is not None:
        body["end_date"] = end_date
    data, _ = self._do_json("POST", "/qai/v1/compute/billing", body)
    return BillingResponse.from_dict(data)


# ---------------------------------------------------------------------------
# Async Client methods
# ---------------------------------------------------------------------------

async def _async_chat_job(
    self: AsyncClient,
    request: ChatRequest,
) -> JobCreateResponse:
    """Submit a chat completion as an async job.

    Useful for long-running models (e.g. Opus) where synchronous
    /qai/v1/chat may time out. Use stream_job() or poll_job() to get the result.
    """
    params = request.to_dict()
    params.pop("stream", None)
    return await self.create_job("chat", params)


async def _async_stream_job(
    self: AsyncClient,
    job_id: str,
) -> AsyncIterator[JobStreamEvent]:
    """Stream job progress via SSE. Yields JobStreamEvent objects.

    Events: "progress" (status update), "complete" (with result), "error".
    """
    url = self.base_url + f"/qai/v1/jobs/{job_id}/stream"
    headers = self._headers()
    headers["Accept"] = "text/event-stream"

    stream_client = httpx.AsyncClient()
    try:
        async with stream_client.stream(
            "GET",
            url,
            headers=headers,
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
                    return
                try:
                    data = json.loads(payload)
                    event = JobStreamEvent.from_dict(data)
                except json.JSONDecodeError as exc:
                    yield JobStreamEvent(type="error", error=f"parse SSE: {exc}")
                    return

                yield event

                if event.type in ("complete", "error"):
                    return
    finally:
        await stream_client.aclose()


async def _async_voice_library(
    self: AsyncClient,
    *,
    query: str | None = None,
    page_size: int | None = None,
    cursor: str | None = None,
    gender: str | None = None,
    language: str | None = None,
    use_case: str | None = None,
) -> SharedVoicesResponse:
    """Browse the shared voice library with optional filters."""
    params: list[str] = []
    if query is not None:
        params.append(f"query={query}")
    if page_size is not None:
        params.append(f"page_size={page_size}")
    if cursor is not None:
        params.append(f"cursor={cursor}")
    if gender is not None:
        params.append(f"gender={gender}")
    if language is not None:
        params.append(f"language={language}")
    if use_case is not None:
        params.append(f"use_case={use_case}")
    path = "/qai/v1/voices/library"
    if params:
        path += "?" + "&".join(params)
    data, _ = await self._do_json("GET", path)
    return SharedVoicesResponse.from_dict(data)


async def _async_add_voice_from_library(
    self: AsyncClient,
    public_owner_id: str,
    voice_id: str,
    *,
    name: str | None = None,
) -> AddVoiceFromLibraryResponse:
    """Add a shared voice from the library to the user's account."""
    body: dict[str, Any] = {
        "public_owner_id": public_owner_id,
        "voice_id": voice_id,
    }
    if name is not None:
        body["name"] = name
    data, _ = await self._do_json("POST", "/qai/v1/voices/library/add", body)
    return AddVoiceFromLibraryResponse.from_dict(data)


async def _async_create_finetune(
    self: AsyncClient,
    name: str,
    samples: list[str],
    *,
    description: str | None = None,
) -> MusicFinetuneInfo:
    """Create a new music finetune from audio samples (base64-encoded)."""
    body: dict[str, Any] = {"name": name, "samples": samples}
    if description is not None:
        body["description"] = description
    data, _ = await self._do_json("POST", "/qai/v1/audio/finetunes", body)
    return MusicFinetuneInfo.from_dict(data)


async def _async_list_finetunes(self: AsyncClient) -> MusicFinetuneListResponse:
    """List all music finetunes for the authenticated user."""
    data, _ = await self._do_json("GET", "/qai/v1/audio/finetunes")
    return MusicFinetuneListResponse.from_dict(data)


async def _async_delete_finetune(self: AsyncClient, finetune_id: str) -> None:
    """Delete a music finetune by ID."""
    await self._do_json("DELETE", f"/qai/v1/audio/finetunes/{finetune_id}")


async def _async_compute_billing(
    self: AsyncClient,
    *,
    instance_id: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
) -> BillingResponse:
    """Query compute billing from BigQuery via the QAI backend."""
    body: dict[str, Any] = {}
    if instance_id is not None:
        body["instance_id"] = instance_id
    if start_date is not None:
        body["start_date"] = start_date
    if end_date is not None:
        body["end_date"] = end_date
    data, _ = await self._do_json("POST", "/qai/v1/compute/billing", body)
    return BillingResponse.from_dict(data)


# ---------------------------------------------------------------------------
# Patch methods onto classes
# ---------------------------------------------------------------------------

Client.chat_job = _sync_chat_job
Client.stream_job = _sync_stream_job
Client.voice_library = _sync_voice_library
Client.add_voice_from_library = _sync_add_voice_from_library
Client.create_finetune = _sync_create_finetune
Client.list_finetunes = _sync_list_finetunes
Client.delete_finetune = _sync_delete_finetune
Client.compute_billing = _sync_compute_billing

AsyncClient.chat_job = _async_chat_job
AsyncClient.stream_job = _async_stream_job
AsyncClient.voice_library = _async_voice_library
AsyncClient.add_voice_from_library = _async_add_voice_from_library
AsyncClient.create_finetune = _async_create_finetune
AsyncClient.list_finetunes = _async_list_finetunes
AsyncClient.delete_finetune = _async_delete_finetune
AsyncClient.compute_billing = _async_compute_billing


# -- 3D Mesh: Remesh, Rig, Animate ---

def _remesh(self, *, input_task_id=None, model_url=None, target_formats=None,
            topology=None, target_polycount=None, resize_height=None,
            origin_at=None, convert_format_only=False):
    """Remesh a 3D model. Submits job and polls to completion."""
    params = {}
    if input_task_id: params["input_task_id"] = input_task_id
    if model_url: params["model_url"] = model_url
    if target_formats: params["target_formats"] = target_formats
    if topology: params["topology"] = topology
    if target_polycount: params["target_polycount"] = target_polycount
    if resize_height: params["resize_height"] = resize_height
    if origin_at: params["origin_at"] = origin_at
    if convert_format_only: params["convert_format_only"] = True
    job = self.create_job("3d/remesh", params)
    return self.poll_job(job["job_id"], interval_ms=5000, max_attempts=120)

def _rig(self, *, input_task_id=None, model_url=None, height_meters=None, texture_image_url=None):
    """Rig a humanoid 3D model. Returns rigged character + basic animations."""
    params = {}
    if input_task_id: params["input_task_id"] = input_task_id
    if model_url: params["model_url"] = model_url
    if height_meters: params["height_meters"] = height_meters
    if texture_image_url: params["texture_image_url"] = texture_image_url
    job = self.create_job("3d/rig", params)
    return self.poll_job(job["job_id"], interval_ms=5000, max_attempts=120)

def _animate(self, rig_task_id, action_id, *, operation_type=None, fps=None):
    """Apply an animation to a rigged character."""
    params = {"rig_task_id": rig_task_id, "action_id": action_id}
    if operation_type:
        pp = {"operation_type": operation_type}
        if fps: pp["fps"] = fps
        params["post_process"] = pp
    job = self.create_job("3d/animate", params)
    return self.poll_job(job["job_id"], interval_ms=5000, max_attempts=120)

async def _remesh_async(self, **kwargs):
    """Async remesh."""
    params = {k: v for k, v in kwargs.items() if v is not None and v is not False}
    job = await self.create_job("3d/remesh", params)
    return await self.poll_job(job["job_id"], interval_ms=5000, max_attempts=120)

async def _rig_async(self, **kwargs):
    """Async rig."""
    params = {k: v for k, v in kwargs.items() if v is not None}
    job = await self.create_job("3d/rig", params)
    return await self.poll_job(job["job_id"], interval_ms=5000, max_attempts=120)

async def _animate_async(self, rig_task_id, action_id, *, operation_type=None, fps=None):
    """Async animate."""
    params = {"rig_task_id": rig_task_id, "action_id": action_id}
    if operation_type:
        pp = {"operation_type": operation_type}
        if fps: pp["fps"] = fps
        params["post_process"] = pp
    job = await self.create_job("3d/animate", params)
    return await self.poll_job(job["job_id"], interval_ms=5000, max_attempts=120)

# Patch onto classes
try:
    from .client import Client, AsyncClient
    Client.remesh = _remesh
    Client.rig = _rig
    Client.animate = _animate
    AsyncClient.remesh = _remesh_async
    AsyncClient.rig = _rig_async
    AsyncClient.animate = _animate_async
except Exception:
    pass

def _retexture(self, *, input_task_id=None, model_url=None, text_style_prompt=None,
               image_style_url=None, ai_model=None, enable_original_uv=None,
               enable_pbr=None, remove_lighting=None, target_formats=None):
    """Retexture a 3D model with AI-generated textures."""
    params = {}
    if input_task_id: params["input_task_id"] = input_task_id
    if model_url: params["model_url"] = model_url
    if text_style_prompt: params["text_style_prompt"] = text_style_prompt
    if image_style_url: params["image_style_url"] = image_style_url
    if ai_model: params["ai_model"] = ai_model
    if enable_original_uv is not None: params["enable_original_uv"] = enable_original_uv
    if enable_pbr is not None: params["enable_pbr"] = enable_pbr
    if remove_lighting is not None: params["remove_lighting"] = remove_lighting
    if target_formats: params["target_formats"] = target_formats
    job = self.create_job("3d/retexture", params)
    return self.poll_job(job["job_id"], interval_ms=5000, max_attempts=120)

async def _retexture_async(self, **kwargs):
    """Async retexture."""
    params = {k: v for k, v in kwargs.items() if v is not None}
    job = await self.create_job("3d/retexture", params)
    return await self.poll_job(job["job_id"], interval_ms=5000, max_attempts=120)

try:
    from .client import Client, AsyncClient
    Client.retexture = _retexture
    AsyncClient.retexture = _retexture_async
except Exception:
    pass

def _realtime_session_with(self, **kwargs):
    """Request a realtime session with full config (voice, prompt, tools for ElevenLabs ConvAI)."""
    data, _ = self._do_json("POST", "/qai/v1/realtime/session", kwargs)
    return data

async def _realtime_session_with_async(self, **kwargs):
    """Async realtime session with full config."""
    data, _ = await self._do_json("POST", "/qai/v1/realtime/session", kwargs)
    return data

try:
    from .client import Client, AsyncClient
    Client.realtime_session_with = _realtime_session_with
    AsyncClient.realtime_session_with = _realtime_session_with_async
except Exception:
    pass
