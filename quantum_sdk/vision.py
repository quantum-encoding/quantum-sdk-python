"""Vision module — image analysis, object detection, OCR, quality assessment."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Request types
# ---------------------------------------------------------------------------

@dataclass
class VisionContext:
    """Domain context for relevance analysis."""

    installation_type: str = ""
    phase: str = ""
    expected_items: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionContext:
        return cls(
            installation_type=data.get("installation_type", ""),
            phase=data.get("phase", ""),
            expected_items=data.get("expected_items", []),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.installation_type:
            d["installation_type"] = self.installation_type
        if self.phase:
            d["phase"] = self.phase
        if self.expected_items:
            d["expected_items"] = self.expected_items
        return d


@dataclass
class VisionRequest:
    """Request body for vision analysis endpoints."""

    image_base64: str = ""
    image_url: str = ""
    model: str = ""
    profile: str = ""
    context: VisionContext | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionRequest:
        ctx = None
        if data.get("context"):
            ctx = VisionContext.from_dict(data["context"])
        return cls(
            image_base64=data.get("image_base64", ""),
            image_url=data.get("image_url", ""),
            model=data.get("model", ""),
            profile=data.get("profile", ""),
            context=ctx,
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.image_base64:
            d["image_base64"] = self.image_base64
        if self.image_url:
            d["image_url"] = self.image_url
        if self.model:
            d["model"] = self.model
        if self.profile:
            d["profile"] = self.profile
        if self.context:
            d["context"] = self.context.to_dict()
        return d


# ---------------------------------------------------------------------------
# Response types
# ---------------------------------------------------------------------------

@dataclass
class DetectedObject:
    """A detected object with bounding box."""

    label: str = ""
    confidence: float = 0.0
    bounding_box: tuple[int, int, int, int] = (0, 0, 0, 0)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DetectedObject:
        bb = data.get("bounding_box", [0, 0, 0, 0])
        return cls(
            label=data.get("label", ""),
            confidence=data.get("confidence", 0.0),
            bounding_box=(
                int(bb[0]) if len(bb) > 0 else 0,
                int(bb[1]) if len(bb) > 1 else 0,
                int(bb[2]) if len(bb) > 2 else 0,
                int(bb[3]) if len(bb) > 3 else 0,
            ),
        )


@dataclass
class QualityAssessment:
    """Image quality assessment."""

    overall: str = ""
    score: float = 0.0
    blur: str = ""
    darkness: str = ""
    resolution: str = ""
    exposure: str = ""
    issues: list[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityAssessment:
        return cls(
            overall=data.get("overall", ""),
            score=data.get("score", 0.0),
            blur=data.get("blur", ""),
            darkness=data.get("darkness", ""),
            resolution=data.get("resolution", ""),
            exposure=data.get("exposure", ""),
            issues=data.get("issues", []),
        )


@dataclass
class RelevanceCheck:
    """Relevance check against expected content."""

    relevant: bool = False
    score: float = 0.0
    expected_items: list[str] = field(default_factory=list)
    found_items: list[str] = field(default_factory=list)
    missing_items: list[str] = field(default_factory=list)
    unexpected_items: list[str] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelevanceCheck:
        return cls(
            relevant=data.get("relevant", False),
            score=data.get("score", 0.0),
            expected_items=data.get("expected_items", []),
            found_items=data.get("found_items", []),
            missing_items=data.get("missing_items", []),
            unexpected_items=data.get("unexpected_items", []),
            notes=data.get("notes", ""),
        )


@dataclass
class TextOverlay:
    """A detected text region in the image."""

    text: str = ""
    bounding_box: tuple[int, int, int, int] | None = None
    overlay_type: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextOverlay:
        bb = data.get("bounding_box")
        parsed_bb = None
        if bb and len(bb) >= 4:
            parsed_bb = (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
        return cls(
            text=data.get("text", ""),
            bounding_box=parsed_bb,
            overlay_type=data.get("type", data.get("overlay_type", "")),
        )


@dataclass
class OcrResult:
    """OCR / text extraction result."""

    text: str = ""
    metadata: dict[str, str] = field(default_factory=dict)
    overlays: list[TextOverlay] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> OcrResult:
        overlays = [TextOverlay.from_dict(o) for o in data.get("overlays", [])]
        return cls(
            text=data.get("text", ""),
            metadata=data.get("metadata", {}),
            overlays=overlays,
        )


@dataclass
class VisionResponse:
    """Full vision analysis response."""

    caption: str = ""
    tags: list[str] = field(default_factory=list)
    objects: list[DetectedObject] = field(default_factory=list)
    quality: QualityAssessment | None = None
    relevance: RelevanceCheck | None = None
    ocr: OcrResult | None = None
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VisionResponse:
        objects = [DetectedObject.from_dict(o) for o in data.get("objects", [])]
        quality = None
        if data.get("quality"):
            quality = QualityAssessment.from_dict(data["quality"])
        relevance = None
        if data.get("relevance"):
            relevance = RelevanceCheck.from_dict(data["relevance"])
        ocr = None
        if data.get("ocr"):
            ocr = OcrResult.from_dict(data["ocr"])
        return cls(
            caption=data.get("caption", ""),
            tags=data.get("tags", []),
            objects=objects,
            quality=quality,
            relevance=relevance,
            ocr=ocr,
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


__all__ = [
    "VisionContext",
    "VisionRequest",
    "DetectedObject",
    "QualityAssessment",
    "RelevanceCheck",
    "TextOverlay",
    "OcrResult",
    "VisionResponse",
]
