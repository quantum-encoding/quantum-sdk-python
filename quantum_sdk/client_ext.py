"""Extensions for Client and AsyncClient — 3D mesh helpers.

Importing this module attaches its helpers to Client and AsyncClient;
quantum_sdk/__init__.py imports it right after client. A helper that cannot
be attached raises at import, so a broken patch fails loudly instead of
leaving the method missing.

Only the 3D mesh helpers (remesh, rig, animate, retexture) and
realtime_session_with are patched on. Everything else (voice_library, the
audio finetunes, chat_job, stream_job, ...) is a real method in client.py,
because patched-on attributes are invisible to type checkers and
autocomplete, which for an SDK is a defect in itself.
"""

from __future__ import annotations

from typing import Any, Callable

from .client import AsyncClient, Client


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
    return self.poll_job(job.job_id, interval=5.0, max_attempts=120)

def _rig(self, *, input_task_id=None, model_url=None, height_meters=None, texture_image_url=None):
    """Rig a humanoid 3D model. Returns rigged character + basic animations."""
    params = {}
    if input_task_id: params["input_task_id"] = input_task_id
    if model_url: params["model_url"] = model_url
    if height_meters: params["height_meters"] = height_meters
    if texture_image_url: params["texture_image_url"] = texture_image_url
    job = self.create_job("3d/rig", params)
    return self.poll_job(job.job_id, interval=5.0, max_attempts=120)

def _animate(self, rig_task_id, action_id, *, operation_type=None, fps=None):
    """Apply an animation to a rigged character."""
    params = {"rig_task_id": rig_task_id, "action_id": action_id}
    if operation_type:
        pp = {"operation_type": operation_type}
        if fps: pp["fps"] = fps
        params["post_process"] = pp
    job = self.create_job("3d/animate", params)
    return self.poll_job(job.job_id, interval=5.0, max_attempts=120)

async def _remesh_async(self, **kwargs):
    """Async remesh."""
    params = {k: v for k, v in kwargs.items() if v is not None and v is not False}
    job = await self.create_job("3d/remesh", params)
    return await self.poll_job(job.job_id, interval=5.0, max_attempts=120)

async def _rig_async(self, **kwargs):
    """Async rig."""
    params = {k: v for k, v in kwargs.items() if v is not None}
    job = await self.create_job("3d/rig", params)
    return await self.poll_job(job.job_id, interval=5.0, max_attempts=120)

async def _animate_async(self, rig_task_id, action_id, *, operation_type=None, fps=None):
    """Async animate."""
    params = {"rig_task_id": rig_task_id, "action_id": action_id}
    if operation_type:
        pp = {"operation_type": operation_type}
        if fps: pp["fps"] = fps
        params["post_process"] = pp
    job = await self.create_job("3d/animate", params)
    return await self.poll_job(job.job_id, interval=5.0, max_attempts=120)

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
    return self.poll_job(job.job_id, interval=5.0, max_attempts=120)

async def _retexture_async(self, **kwargs):
    """Async retexture."""
    params = {k: v for k, v in kwargs.items() if v is not None}
    job = await self.create_job("3d/retexture", params)
    return await self.poll_job(job.job_id, interval=5.0, max_attempts=120)

def _realtime_session_with(self, **kwargs):
    """Request a realtime session with full config (voice, prompt, tools for ElevenLabs ConvAI)."""
    data, _ = self._do_json("POST", "/qai/v1/realtime/session", kwargs)
    return data

async def _realtime_session_with_async(self, **kwargs):
    """Async realtime session with full config."""
    data, _ = await self._do_json("POST", "/qai/v1/realtime/session", kwargs)
    return data



def _install(patches: dict[type, dict[str, Callable[..., Any]]]) -> None:
    """Attach each helper to its class, refusing to shadow a method defined
    anywhere else.

    Every name is checked before any is set, so a refused patch leaves the
    classes untouched. The RuntimeError propagates out of the import.
    """
    for cls, methods in patches.items():
        for name in methods:
            existing = getattr(cls, name, None)
            if existing is not None and getattr(existing, "__module__", None) != __name__:
                raise RuntimeError(
                    f"client_ext cannot attach {cls.__name__}.{name}: it would replace "
                    f"the one defined in {getattr(existing, '__module__', '?')}"
                )
    for cls, methods in patches.items():
        for name, fn in methods.items():
            setattr(cls, name, fn)


PATCHES: dict[type, dict[str, Callable[..., Any]]] = {
    Client: {
        "remesh": _remesh,
        "rig": _rig,
        "animate": _animate,
        "retexture": _retexture,
        "realtime_session_with": _realtime_session_with,
    },
    AsyncClient: {
        "remesh": _remesh_async,
        "rig": _rig_async,
        "animate": _animate_async,
        "retexture": _retexture_async,
        "realtime_session_with": _realtime_session_with_async,
    },
}

_install(PATCHES)
