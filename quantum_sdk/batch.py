"""Batch processing — submit multiple prompts in a single request."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── Types ──────────────────────────────────────────────────────────────

@dataclass
class BatchJobInput:
    """A single job in a batch submission."""

    model: str = ""
    prompt: str = ""
    title: str | None = None
    system_prompt: str | None = None
    max_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "prompt": self.prompt}
        if self.title is not None:
            d["title"] = self.title
        if self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        return d


@dataclass
class BatchSubmitResponse:
    """Response from batch submission."""

    job_ids: list[str] = field(default_factory=list)
    status: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchSubmitResponse:
        return cls(job_ids=data.get("job_ids", []), status=data.get("status", ""))


@dataclass
class BatchJsonlResponse:
    """Response from JSONL batch submission."""

    job_ids: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchJsonlResponse:
        return cls(job_ids=data.get("job_ids", []))


@dataclass
class BatchJobInfo:
    """A single job in the batch jobs list."""

    job_id: str = ""
    status: str = ""
    model: str | None = None
    title: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    result: Any = None
    error: str | None = None
    cost_ticks: int = 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchJobInfo:
        return cls(
            job_id=data.get("job_id", ""),
            status=data.get("status", ""),
            model=data.get("model"),
            title=data.get("title"),
            created_at=data.get("created_at"),
            completed_at=data.get("completed_at"),
            result=data.get("result"),
            error=data.get("error"),
            cost_ticks=data.get("cost_ticks", 0),
        )


@dataclass
class BatchJobsResponse:
    """Response from listing batch jobs."""

    jobs: list[BatchJobInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BatchJobsResponse:
        return cls(jobs=[BatchJobInfo.from_dict(j) for j in data.get("jobs", [])])


# ── Functions (use with Client or AsyncClient) ─────────────────────────

def batch_submit_sync(client: Any, jobs: list[BatchJobInput]) -> BatchSubmitResponse:
    """Submit a batch of jobs (sync)."""
    body = {"jobs": [j.to_dict() for j in jobs]}
    data, _ = client._do_json("POST", "/qai/v1/batch", body)
    return BatchSubmitResponse.from_dict(data)


def batch_submit_jsonl_sync(client: Any, jsonl: str) -> BatchJsonlResponse:
    """Submit a batch via JSONL (sync)."""
    data, _ = client._do_json("POST", "/qai/v1/batch/jsonl", {"jsonl": jsonl})
    return BatchJsonlResponse.from_dict(data)


def batch_jobs_sync(client: Any) -> BatchJobsResponse:
    """List all batch jobs (sync)."""
    data, _ = client._do_json("GET", "/qai/v1/batch/jobs")
    return BatchJobsResponse.from_dict(data)


def batch_job_sync(client: Any, job_id: str) -> BatchJobInfo:
    """Get a single batch job (sync)."""
    data, _ = client._do_json("GET", f"/qai/v1/batch/jobs/{job_id}")
    return BatchJobInfo.from_dict(data)


async def batch_submit_async(client: Any, jobs: list[BatchJobInput]) -> BatchSubmitResponse:
    """Submit a batch of jobs (async)."""
    body = {"jobs": [j.to_dict() for j in jobs]}
    data, _ = await client._do_json("POST", "/qai/v1/batch", body)
    return BatchSubmitResponse.from_dict(data)


async def batch_submit_jsonl_async(client: Any, jsonl: str) -> BatchJsonlResponse:
    """Submit a batch via JSONL (async)."""
    data, _ = await client._do_json("POST", "/qai/v1/batch/jsonl", {"jsonl": jsonl})
    return BatchJsonlResponse.from_dict(data)


async def batch_jobs_async(client: Any) -> BatchJobsResponse:
    """List all batch jobs (async)."""
    data, _ = await client._do_json("GET", "/qai/v1/batch/jobs")
    return BatchJobsResponse.from_dict(data)


async def batch_job_async(client: Any, job_id: str) -> BatchJobInfo:
    """Get a single batch job (async)."""
    data, _ = await client._do_json("GET", f"/qai/v1/batch/jobs/{job_id}")
    return BatchJobInfo.from_dict(data)
