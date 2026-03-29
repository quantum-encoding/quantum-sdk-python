"""Missions module — multi-agent orchestration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------

@dataclass
class MissionWorkerConfig:
    """Worker configuration within a mission."""

    model: str = ""
    tier: str = ""
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionWorkerConfig:
        return cls(
            model=data.get("model", ""),
            tier=data.get("tier", ""),
            description=data.get("description", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model}
        if self.tier:
            d["tier"] = self.tier
        if self.description:
            d["description"] = self.description
        return d


@dataclass
class MissionCreateRequest:
    """Request body for creating a mission."""

    goal: str = ""
    strategy: str = ""
    conductor_model: str = ""
    workers: dict[str, MissionWorkerConfig] | None = None
    max_steps: int = 0
    system_prompt: str = ""
    session_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionCreateRequest:
        workers = None
        raw = data.get("workers")
        if raw and isinstance(raw, dict):
            workers = {k: MissionWorkerConfig.from_dict(v) for k, v in raw.items()}
        return cls(
            goal=data.get("goal", ""),
            strategy=data.get("strategy", ""),
            conductor_model=data.get("conductor_model", ""),
            workers=workers,
            max_steps=data.get("max_steps", 0),
            system_prompt=data.get("system_prompt", ""),
            session_id=data.get("session_id", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"goal": self.goal}
        if self.strategy:
            d["strategy"] = self.strategy
        if self.conductor_model:
            d["conductor_model"] = self.conductor_model
        if self.workers:
            d["workers"] = {k: v.to_dict() for k, v in self.workers.items()}
        if self.max_steps:
            d["max_steps"] = self.max_steps
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        if self.session_id:
            d["session_id"] = self.session_id
        return d


@dataclass
class MissionChatRequest:
    """Request body for chatting with a mission's architect."""

    message: str = ""
    stream: bool | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionChatRequest:
        return cls(
            message=data.get("message", ""),
            stream=data.get("stream"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"message": self.message}
        if self.stream is not None:
            d["stream"] = self.stream
        return d


@dataclass
class MissionPlanUpdate:
    """Request body for updating a mission plan."""

    tasks: list[dict[str, Any]] | None = None
    workers: dict[str, MissionWorkerConfig] | None = None
    system_prompt: str = ""
    max_steps: int = 0
    context: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionPlanUpdate:
        workers = None
        raw = data.get("workers")
        if raw and isinstance(raw, dict):
            workers = {k: MissionWorkerConfig.from_dict(v) for k, v in raw.items()}
        return cls(
            tasks=data.get("tasks"),
            workers=workers,
            system_prompt=data.get("system_prompt", ""),
            max_steps=data.get("max_steps", 0),
            context=data.get("context", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.tasks is not None:
            d["tasks"] = self.tasks
        if self.workers:
            d["workers"] = {k: v.to_dict() for k, v in self.workers.items()}
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        if self.max_steps:
            d["max_steps"] = self.max_steps
        if self.context:
            d["context"] = self.context
        return d


@dataclass
class MissionConfirmStructure:
    """Request body for confirming/rejecting a mission structure."""

    confirmed: bool = False
    feedback: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionConfirmStructure:
        return cls(
            confirmed=data.get("confirmed", False),
            feedback=data.get("feedback", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"confirmed": self.confirmed}
        if self.feedback:
            d["feedback"] = self.feedback
        return d


@dataclass
class MissionApproveRequest:
    """Request body for approving a completed mission."""

    commit_sha: str = ""
    comment: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionApproveRequest:
        return cls(
            commit_sha=data.get("commit_sha", ""),
            comment=data.get("comment", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.commit_sha:
            d["commit_sha"] = self.commit_sha
        if self.comment:
            d["comment"] = self.comment
        return d


@dataclass
class MissionImportRequest:
    """Request body for importing a plan as a new mission."""

    goal: str = ""
    strategy: str = ""
    conductor_model: str = ""
    workers: dict[str, MissionWorkerConfig] | None = None
    tasks: list[dict[str, Any]] = field(default_factory=list)
    system_prompt: str = ""
    max_steps: int = 0
    auto_execute: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionImportRequest:
        workers = None
        raw = data.get("workers")
        if raw and isinstance(raw, dict):
            workers = {k: MissionWorkerConfig.from_dict(v) for k, v in raw.items()}
        return cls(
            goal=data.get("goal", ""),
            strategy=data.get("strategy", ""),
            conductor_model=data.get("conductor_model", ""),
            workers=workers,
            tasks=data.get("tasks", []),
            system_prompt=data.get("system_prompt", ""),
            max_steps=data.get("max_steps", 0),
            auto_execute=data.get("auto_execute", False),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"goal": self.goal}
        if self.strategy:
            d["strategy"] = self.strategy
        if self.conductor_model:
            d["conductor_model"] = self.conductor_model
        if self.workers:
            d["workers"] = {k: v.to_dict() for k, v in self.workers.items()}
        if self.tasks:
            d["tasks"] = self.tasks
        if self.system_prompt:
            d["system_prompt"] = self.system_prompt
        if self.max_steps:
            d["max_steps"] = self.max_steps
        if self.auto_execute:
            d["auto_execute"] = self.auto_execute
        return d


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class MissionCreateResponse:
    """Response from mission creation."""

    mission_id: str = ""
    status: str = ""
    session_id: str = ""
    conductor_model: str = ""
    strategy: str = ""
    workers: dict[str, MissionWorkerConfig] | None = None
    created_at: str = ""
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionCreateResponse:
        workers = None
        raw = data.get("workers")
        if raw and isinstance(raw, dict):
            workers = {k: MissionWorkerConfig.from_dict(v) for k, v in raw.items()}
        return cls(
            mission_id=data.get("mission_id", ""),
            status=data.get("status", ""),
            session_id=data.get("session_id", ""),
            conductor_model=data.get("conductor_model", ""),
            strategy=data.get("strategy", ""),
            workers=workers,
            created_at=data.get("created_at", ""),
            request_id=data.get("request_id", ""),
        )


@dataclass
class MissionTask:
    """A task within a mission."""

    id: str = ""
    name: str = ""
    description: str = ""
    worker: str = ""
    model: str = ""
    status: str = ""
    result: str = ""
    error: str = ""
    step: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionTask:
        return cls(
            id=data.get("id", ""),
            name=data.get("name", ""),
            description=data.get("description", ""),
            worker=data.get("worker", ""),
            model=data.get("model", ""),
            status=data.get("status", ""),
            result=data.get("result", ""),
            error=data.get("error", ""),
            step=data.get("step", 0),
            tokens_in=data.get("tokens_in", 0),
            tokens_out=data.get("tokens_out", 0),
        )


@dataclass
class MissionDetail:
    """Mission detail (from GET /missions/{id})."""

    id: str = ""
    user_id: str = ""
    goal: str = ""
    strategy: str = ""
    conductor_model: str = ""
    status: str = ""
    created_at: str = ""
    started_at: str = ""
    completed_at: str = ""
    error: str = ""
    cost_ticks: int = 0
    total_steps: int = 0
    session_id: str = ""
    result: str = ""
    tasks: list[MissionTask] = field(default_factory=list)
    approved: bool = False
    commit_sha: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionDetail:
        tasks = [MissionTask.from_dict(t) for t in data.get("tasks", [])]
        return cls(
            id=data.get("id", ""),
            user_id=data.get("user_id", ""),
            goal=data.get("goal", ""),
            strategy=data.get("strategy", ""),
            conductor_model=data.get("conductor_model", ""),
            status=data.get("status", ""),
            created_at=data.get("created_at", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            error=data.get("error", ""),
            cost_ticks=data.get("cost_ticks", 0),
            total_steps=data.get("total_steps", 0),
            session_id=data.get("session_id", ""),
            result=data.get("result", ""),
            tasks=tasks,
            approved=data.get("approved", False),
            commit_sha=data.get("commit_sha", ""),
        )


@dataclass
class MissionListResponse:
    """Response from listing missions."""

    missions: list[MissionDetail] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionListResponse:
        missions = [MissionDetail.from_dict(m) for m in data.get("missions", [])]
        return cls(missions=missions)


@dataclass
class MissionChatUsage:
    """Token usage for a mission chat response."""

    input_tokens: int = 0
    output_tokens: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionChatUsage:
        return cls(
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
        )


@dataclass
class MissionChatResponse:
    """Response from chatting with the architect."""

    mission_id: str = ""
    content: str = ""
    model: str = ""
    cost_ticks: int = 0
    usage: MissionChatUsage | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionChatResponse:
        usage = None
        if data.get("usage"):
            usage = MissionChatUsage.from_dict(data["usage"])
        return cls(
            mission_id=data.get("mission_id", ""),
            content=data.get("content", ""),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            usage=usage,
        )


@dataclass
class MissionCheckpoint:
    """A git checkpoint within a mission."""

    id: str = ""
    commit_sha: str = ""
    message: str = ""
    created_at: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionCheckpoint:
        return cls(
            id=data.get("id", ""),
            commit_sha=data.get("commit_sha", ""),
            message=data.get("message", ""),
            created_at=data.get("created_at", ""),
        )


@dataclass
class MissionCheckpointsResponse:
    """Response from listing checkpoints."""

    mission_id: str = ""
    checkpoints: list[MissionCheckpoint] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionCheckpointsResponse:
        checkpoints = [MissionCheckpoint.from_dict(c) for c in data.get("checkpoints", [])]
        return cls(
            mission_id=data.get("mission_id", ""),
            checkpoints=checkpoints,
        )


@dataclass
class MissionStatusResponse:
    """Generic status response for mission operations."""

    mission_id: str = ""
    status: str = ""
    confirmed: bool | None = None
    approved: bool | None = None
    deleted: bool | None = None
    updated: bool | None = None
    commit_sha: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MissionStatusResponse:
        return cls(
            mission_id=data.get("mission_id", ""),
            status=data.get("status", ""),
            confirmed=data.get("confirmed"),
            approved=data.get("approved"),
            deleted=data.get("deleted"),
            updated=data.get("updated"),
            commit_sha=data.get("commit_sha", ""),
        )


__all__ = [
    "MissionWorkerConfig",
    "MissionCreateRequest",
    "MissionChatRequest",
    "MissionPlanUpdate",
    "MissionConfirmStructure",
    "MissionApproveRequest",
    "MissionImportRequest",
    "MissionCreateResponse",
    "MissionTask",
    "MissionDetail",
    "MissionListResponse",
    "MissionChatUsage",
    "MissionChatResponse",
    "MissionCheckpoint",
    "MissionCheckpointsResponse",
    "MissionStatusResponse",
]
