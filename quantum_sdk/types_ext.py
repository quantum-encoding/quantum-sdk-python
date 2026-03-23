"""Extended types for Quantum AI API — voice library, finetunes, billing, job streaming."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Voice Library (shared/community voices)
# ---------------------------------------------------------------------------

@dataclass
class SharedVoice:
    """A voice from the shared voice library."""

    public_owner_id: str = ""
    voice_id: str = ""
    name: str = ""
    category: str = ""
    description: str = ""
    preview_url: str = ""
    gender: str = ""
    age: str = ""
    accent: str = ""
    language: str = ""
    use_case: str = ""
    rate: float = 0.0
    cloned_by_count: int = 0
    free_users_allowed: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedVoice:
        return cls(
            public_owner_id=data.get("public_owner_id", ""),
            voice_id=data.get("voice_id", ""),
            name=data.get("name", ""),
            category=data.get("category", ""),
            description=data.get("description", ""),
            preview_url=data.get("preview_url", ""),
            gender=data.get("gender", ""),
            age=data.get("age", ""),
            accent=data.get("accent", ""),
            language=data.get("language", ""),
            use_case=data.get("use_case", ""),
            rate=data.get("rate", 0.0),
            cloned_by_count=data.get("cloned_by_count", 0),
            free_users_allowed=data.get("free_users_allowed", False),
        )


@dataclass
class SharedVoicesResponse:
    """Response from browsing the voice library."""

    voices: list[SharedVoice] = field(default_factory=list)
    next_cursor: str = ""
    has_more: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SharedVoicesResponse:
        voices = [SharedVoice.from_dict(v) for v in data.get("voices", [])]
        return cls(
            voices=voices,
            next_cursor=data.get("next_cursor", ""),
            has_more=data.get("has_more", False),
        )


@dataclass
class AddVoiceFromLibraryResponse:
    """Response from adding a shared voice."""

    voice_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AddVoiceFromLibraryResponse:
        return cls(voice_id=data.get("voice_id", ""))


# ---------------------------------------------------------------------------
# Music Finetunes
# ---------------------------------------------------------------------------

@dataclass
class MusicFinetuneInfo:
    """Information about a music finetune."""

    finetune_id: str = ""
    name: str = ""
    description: str = ""
    status: str = ""
    model_id: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MusicFinetuneInfo:
        return cls(
            finetune_id=data.get("finetune_id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            status=data.get("status", ""),
            model_id=data.get("model_id", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass
class MusicFinetuneListResponse:
    """Response from listing music finetunes."""

    finetunes: list[MusicFinetuneInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MusicFinetuneListResponse:
        finetunes = [MusicFinetuneInfo.from_dict(f) for f in data.get("finetunes", [])]
        return cls(finetunes=finetunes)


# ---------------------------------------------------------------------------
# Compute Billing
# ---------------------------------------------------------------------------

@dataclass
class BillingEntry:
    """A single billing line item from BigQuery."""

    instance_id: str = ""
    instance_name: str = ""
    cost_usd: float = 0.0
    usage_hours: float = 0.0
    sku_description: str = ""
    start_time: str = ""
    end_time: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BillingEntry:
        return cls(
            instance_id=data.get("instance_id", ""),
            instance_name=data.get("instance_name", ""),
            cost_usd=data.get("cost_usd", 0.0),
            usage_hours=data.get("usage_hours", 0.0),
            sku_description=data.get("sku_description", ""),
            start_time=data.get("start_time", ""),
            end_time=data.get("end_time", ""),
        )


@dataclass
class BillingResponse:
    """Response from a compute billing query."""

    entries: list[BillingEntry] = field(default_factory=list)
    total_cost_usd: float = 0.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BillingResponse:
        entries = [BillingEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(
            entries=entries,
            total_cost_usd=data.get("total_cost_usd", 0.0),
        )


# ---------------------------------------------------------------------------
# Job Stream Events
# ---------------------------------------------------------------------------

@dataclass
class JobStreamEvent:
    """A single event from an SSE job stream."""

    type: str = ""
    job_id: str = ""
    status: str = ""
    result: dict[str, Any] | None = None
    error: str = ""
    cost_ticks: int = 0
    completed_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobStreamEvent:
        return cls(
            type=data.get("type", ""),
            job_id=data.get("job_id", ""),
            status=data.get("status", ""),
            result=data.get("result"),
            error=data.get("error", ""),
            cost_ticks=data.get("cost_ticks", 0),
            completed_at=data.get("completed_at", ""),
        )
