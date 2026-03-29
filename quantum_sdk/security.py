"""Security module — prompt injection scanning and threat registry."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------

@dataclass
class SecurityScanUrlRequest:
    """Request body for scanning a URL for prompt injection."""

    url: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityScanUrlRequest:
        return cls(url=data.get("url", ""))

    def to_dict(self) -> dict[str, Any]:
        return {"url": self.url}


@dataclass
class SecurityScanHtmlRequest:
    """Request body for scanning raw HTML content."""

    html: str = ""
    visible_text: str = ""
    url: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityScanHtmlRequest:
        return cls(
            html=data.get("html", ""),
            visible_text=data.get("visible_text", ""),
            url=data.get("url", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"html": self.html}
        if self.visible_text:
            d["visible_text"] = self.visible_text
        if self.url:
            d["url"] = self.url
        return d


@dataclass
class SecurityReportRequest:
    """Request body for reporting a suspicious URL."""

    url: str = ""
    description: str = ""
    category: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityReportRequest:
        return cls(
            url=data.get("url", ""),
            description=data.get("description", ""),
            category=data.get("category", ""),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"url": self.url}
        if self.description:
            d["description"] = self.description
        if self.category:
            d["category"] = self.category
        return d


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class SecurityFinding:
    """A single detected injection pattern."""

    category: str = ""
    pattern: str = ""
    content: str = ""
    location: str = ""
    threat: str = ""
    confidence: float = 0.0
    description: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityFinding:
        return cls(
            category=data.get("category", ""),
            pattern=data.get("pattern", ""),
            content=data.get("content", ""),
            location=data.get("location", ""),
            threat=data.get("threat", ""),
            confidence=data.get("confidence", 0.0),
            description=data.get("description", ""),
        )


@dataclass
class SecurityAssessment:
    """Threat assessment for a scanned page."""

    url: str = ""
    threat_level: str = ""
    threat_score: float = 0.0
    findings: list[SecurityFinding] = field(default_factory=list)
    hidden_text_length: int = 0
    visible_text_length: int = 0
    hidden_ratio: float = 0.0
    summary: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityAssessment:
        findings = [SecurityFinding.from_dict(f) for f in data.get("findings", [])]
        return cls(
            url=data.get("url", ""),
            threat_level=data.get("threat_level", ""),
            threat_score=data.get("threat_score", 0.0),
            findings=findings,
            hidden_text_length=data.get("hidden_text_length", 0),
            visible_text_length=data.get("visible_text_length", 0),
            hidden_ratio=data.get("hidden_ratio", 0.0),
            summary=data.get("summary", ""),
        )


@dataclass
class SecurityScanResponse:
    """Response from a security scan."""

    assessment: SecurityAssessment = field(default_factory=SecurityAssessment)
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityScanResponse:
        assessment = SecurityAssessment()
        if data.get("assessment"):
            assessment = SecurityAssessment.from_dict(data["assessment"])
        return cls(
            assessment=assessment,
            request_id=data.get("request_id", ""),
        )


@dataclass
class SecurityCheckResponse:
    """Response from checking a URL against the registry."""

    url: str = ""
    blocked: bool = False
    threat_level: str = ""
    threat_score: float | None = None
    categories: list[str] | None = None
    first_seen: str = ""
    last_seen: str = ""
    report_count: int | None = None
    status: str = ""
    message: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityCheckResponse:
        return cls(
            url=data.get("url", ""),
            blocked=data.get("blocked", False),
            threat_level=data.get("threat_level", ""),
            threat_score=data.get("threat_score"),
            categories=data.get("categories"),
            first_seen=data.get("first_seen", ""),
            last_seen=data.get("last_seen", ""),
            report_count=data.get("report_count"),
            status=data.get("status", ""),
            message=data.get("message", ""),
        )


@dataclass
class SecurityBlocklistEntry:
    """A single blocklist entry."""

    id: str = ""
    url: str = ""
    status: str = ""
    threat_level: str = ""
    threat_score: float = 0.0
    categories: list[str] = field(default_factory=list)
    findings_count: int = 0
    hidden_ratio: float = 0.0
    first_seen: str = ""
    summary: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityBlocklistEntry:
        return cls(
            id=data.get("id", ""),
            url=data.get("url", ""),
            status=data.get("status", ""),
            threat_level=data.get("threat_level", ""),
            threat_score=data.get("threat_score", 0.0),
            categories=data.get("categories", []),
            findings_count=data.get("findings_count", 0),
            hidden_ratio=data.get("hidden_ratio", 0.0),
            first_seen=data.get("first_seen", ""),
            summary=data.get("summary", ""),
        )


@dataclass
class SecurityBlocklistResponse:
    """Response from the blocklist feed."""

    entries: list[SecurityBlocklistEntry] = field(default_factory=list)
    count: int = 0
    status: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityBlocklistResponse:
        entries = [SecurityBlocklistEntry.from_dict(e) for e in data.get("entries", [])]
        return cls(
            entries=entries,
            count=data.get("count", 0),
            status=data.get("status", ""),
        )


@dataclass
class SecurityReportResponse:
    """Response from reporting a URL."""

    url: str = ""
    status: str = ""
    message: str = ""
    threat_level: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SecurityReportResponse:
        return cls(
            url=data.get("url", ""),
            status=data.get("status", ""),
            message=data.get("message", ""),
            threat_level=data.get("threat_level", ""),
        )


__all__ = [
    "SecurityScanUrlRequest",
    "SecurityScanHtmlRequest",
    "SecurityReportRequest",
    "SecurityFinding",
    "SecurityAssessment",
    "SecurityScanResponse",
    "SecurityCheckResponse",
    "SecurityBlocklistEntry",
    "SecurityBlocklistResponse",
    "SecurityReportResponse",
]
