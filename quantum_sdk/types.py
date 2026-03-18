"""Quantum AI API data types — all dataclasses with type hints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

@dataclass
class ChatMessage:
    """A single message in a conversation."""

    role: str
    content: str = ""
    content_blocks: list[ContentBlock] | None = None
    tool_call_id: str | None = None
    is_error: bool = False

    @classmethod
    def user(cls, content: str) -> ChatMessage:
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str) -> ChatMessage:
        return cls(role="assistant", content=content)

    @classmethod
    def system(cls, content: str) -> ChatMessage:
        return cls(role="system", content=content)

    @classmethod
    def tool(cls, tool_call_id: str, content: str, *, is_error: bool = False) -> ChatMessage:
        return cls(role="tool", content=content, tool_call_id=tool_call_id, is_error=is_error)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"role": self.role}
        if self.content:
            d["content"] = self.content
        if self.content_blocks is not None:
            d["content_blocks"] = [b.to_dict() for b in self.content_blocks]
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        if self.is_error:
            d["is_error"] = True
        return d


@dataclass
class ContentBlock:
    """A single block in the response content array."""

    type: str
    text: str = ""
    id: str = ""
    name: str = ""
    input: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"type": self.type}
        if self.text:
            d["text"] = self.text
        if self.id:
            d["id"] = self.id
        if self.name:
            d["name"] = self.name
        if self.input is not None:
            d["input"] = self.input
        return d


@dataclass
class ChatTool:
    """Defines a function the model can call."""

    name: str
    description: str
    parameters: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "description": self.description}
        if self.parameters is not None:
            d["parameters"] = self.parameters
        return d


@dataclass
class ChatUsage:
    """Token counts and cost for a chat response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cost_ticks: int = 0


@dataclass
class ChatRequest:
    """Request body for text generation."""

    model: str
    messages: list[ChatMessage]
    tools: list[ChatTool] | None = None
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    provider_options: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in self.messages],
        }
        if self.tools:
            d["tools"] = [t.to_dict() for t in self.tools]
        if self.stream:
            d["stream"] = True
        if self.temperature is not None:
            d["temperature"] = self.temperature
        if self.max_tokens is not None:
            d["max_tokens"] = self.max_tokens
        if self.provider_options is not None:
            d["provider_options"] = self.provider_options
        return d


@dataclass
class ChatResponse:
    """Response from a non-streaming chat request."""

    id: str = ""
    model: str = ""
    content: list[ContentBlock] = field(default_factory=list)
    usage: ChatUsage | None = None
    stop_reason: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    def text(self) -> str:
        """Concatenated text content, ignoring thinking and tool_use blocks."""
        return "".join(b.text for b in self.content if b.type == "text")

    def thinking(self) -> str:
        """Concatenated thinking content."""
        return "".join(b.text for b in self.content if b.type == "thinking")

    def tool_calls(self) -> list[ContentBlock]:
        """All tool_use blocks from the response."""
        return [b for b in self.content if b.type == "tool_use"]

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChatResponse:
        content = [
            ContentBlock(
                type=b.get("type", ""),
                text=b.get("text", ""),
                id=b.get("id", ""),
                name=b.get("name", ""),
                input=b.get("input"),
            )
            for b in data.get("content", [])
        ]
        usage_data = data.get("usage")
        usage = ChatUsage(
            input_tokens=usage_data.get("input_tokens", 0),
            output_tokens=usage_data.get("output_tokens", 0),
            cost_ticks=usage_data.get("cost_ticks", 0),
        ) if usage_data else None
        return cls(
            id=data.get("id", ""),
            model=data.get("model", ""),
            content=content,
            usage=usage,
            stop_reason=data.get("stop_reason", ""),
        )


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

@dataclass
class StreamDelta:
    """Incremental text in a streaming event."""

    text: str = ""


@dataclass
class StreamToolUse:
    """Tool call from a streaming event."""

    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class StreamEvent:
    """A single event from an SSE chat stream."""

    type: str = ""
    delta: StreamDelta | None = None
    tool_use: StreamToolUse | None = None
    usage: ChatUsage | None = None
    error: str = ""
    done: bool = False


# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------

@dataclass
class ImageRequest:
    """Request body for image generation."""

    model: str
    prompt: str
    count: int | None = None
    size: str | None = None
    aspect_ratio: str | None = None
    quality: str | None = None
    output_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "prompt": self.prompt}
        if self.count is not None:
            d["count"] = self.count
        if self.size is not None:
            d["size"] = self.size
        if self.aspect_ratio is not None:
            d["aspect_ratio"] = self.aspect_ratio
        if self.quality is not None:
            d["quality"] = self.quality
        if self.output_format is not None:
            d["output_format"] = self.output_format
        return d


@dataclass
class GeneratedImage:
    """A single generated image."""

    base64: str = ""
    format: str = ""
    index: int = 0


@dataclass
class ImageResponse:
    """Response from image generation."""

    images: list[GeneratedImage] = field(default_factory=list)
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ImageResponse:
        images = [
            GeneratedImage(
                base64=img.get("base64", ""),
                format=img.get("format", ""),
                index=img.get("index", 0),
            )
            for img in data.get("images", [])
        ]
        return cls(
            images=images,
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class ImageEditRequest:
    """Request body for image editing."""

    model: str
    prompt: str
    input_images: list[str]
    count: int | None = None
    size: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "model": self.model,
            "prompt": self.prompt,
            "input_images": self.input_images,
        }
        if self.count is not None:
            d["count"] = self.count
        if self.size is not None:
            d["size"] = self.size
        return d


# ImageEditResponse is the same shape as ImageResponse
ImageEditResponse = ImageResponse


# ---------------------------------------------------------------------------
# Video
# ---------------------------------------------------------------------------

@dataclass
class VideoRequest:
    """Request body for video generation."""

    model: str
    prompt: str
    duration_seconds: int | None = None
    aspect_ratio: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "prompt": self.prompt}
        if self.duration_seconds is not None:
            d["duration_seconds"] = self.duration_seconds
        if self.aspect_ratio is not None:
            d["aspect_ratio"] = self.aspect_ratio
        return d


@dataclass
class GeneratedVideo:
    """A single generated video."""

    base64: str = ""
    format: str = ""
    size_bytes: int = 0
    index: int = 0


@dataclass
class VideoResponse:
    """Response from video generation."""

    videos: list[GeneratedVideo] = field(default_factory=list)
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoResponse:
        videos = [
            GeneratedVideo(
                base64=v.get("base64", ""),
                format=v.get("format", ""),
                size_bytes=v.get("size_bytes", 0),
                index=v.get("index", 0),
            )
            for v in data.get("videos", [])
        ]
        return cls(
            videos=videos,
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

@dataclass
class TTSRequest:
    """Request body for text-to-speech."""

    model: str
    text: str
    voice: str | None = None
    output_format: str | None = None
    speed: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "text": self.text}
        if self.voice is not None:
            d["voice"] = self.voice
        if self.output_format is not None:
            d["format"] = self.output_format
        if self.speed is not None:
            d["speed"] = self.speed
        return d


@dataclass
class TTSResponse:
    """Response from text-to-speech."""

    audio_base64: str = ""
    format: str = ""
    size_bytes: int = 0
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TTSResponse:
        return cls(
            audio_base64=data.get("audio_base64", ""),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class STTRequest:
    """Request body for speech-to-text."""

    model: str
    audio_base64: str
    filename: str | None = None
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "audio_base64": self.audio_base64}
        if self.filename is not None:
            d["filename"] = self.filename
        if self.language is not None:
            d["language"] = self.language
        return d


@dataclass
class STTResponse:
    """Response from speech-to-text."""

    text: str = ""
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> STTResponse:
        return cls(
            text=data.get("text", ""),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class MusicRequest:
    """Request body for music generation."""

    model: str
    prompt: str
    duration_seconds: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "prompt": self.prompt}
        if self.duration_seconds is not None:
            d["duration_seconds"] = self.duration_seconds
        return d


@dataclass
class MusicClip:
    """A single generated music clip."""

    base64: str = ""
    format: str = ""
    size_bytes: int = 0
    index: int = 0


@dataclass
class MusicResponse:
    """Response from music generation."""

    audio_clips: list[MusicClip] = field(default_factory=list)
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MusicResponse:
        clips = [
            MusicClip(
                base64=c.get("base64", ""),
                format=c.get("format", ""),
                size_bytes=c.get("size_bytes", 0),
                index=c.get("index", 0),
            )
            for c in data.get("audio_clips", [])
        ]
        return cls(
            audio_clips=clips,
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------

@dataclass
class EmbedRequest:
    """Request body for text embeddings."""

    model: str
    input: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "input": self.input}


@dataclass
class EmbedResponse:
    """Response from text embedding."""

    embeddings: list[list[float]] = field(default_factory=list)
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbedResponse:
        return cls(
            embeddings=data.get("embeddings", []),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------

@dataclass
class DocumentRequest:
    """Request body for document extraction."""

    file_base64: str
    filename: str
    output_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"file_base64": self.file_base64, "filename": self.filename}
        if self.output_format is not None:
            d["output_format"] = self.output_format
        return d


@dataclass
class DocumentResponse:
    """Response from document extraction."""

    content: str = ""
    format: str = ""
    meta: dict[str, Any] | None = None
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DocumentResponse:
        return cls(
            content=data.get("content", ""),
            format=data.get("format", ""),
            meta=data.get("meta"),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# RAG
# ---------------------------------------------------------------------------

@dataclass
class RAGSearchRequest:
    """Request body for Vertex AI RAG search."""

    query: str
    corpus: str | None = None
    top_k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"query": self.query}
        if self.corpus is not None:
            d["corpus"] = self.corpus
        if self.top_k is not None:
            d["top_k"] = self.top_k
        return d


@dataclass
class RAGResult:
    """A single result from RAG search."""

    source_uri: str = ""
    source_name: str = ""
    text: str = ""
    score: float = 0.0
    distance: float = 0.0


@dataclass
class RAGSearchResponse:
    """Response from RAG search."""

    results: list[RAGResult] = field(default_factory=list)
    query: str = ""
    corpora: list[str] | None = None
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RAGSearchResponse:
        results = [
            RAGResult(
                source_uri=r.get("source_uri", ""),
                source_name=r.get("source_name", ""),
                text=r.get("text", ""),
                score=r.get("score", 0.0),
                distance=r.get("distance", 0.0),
            )
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            query=data.get("query", ""),
            corpora=data.get("corpora"),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class RAGCorpus:
    """An available RAG corpus."""

    name: str = ""
    display_name: str = ""
    description: str = ""
    state: str = ""


@dataclass
class SurrealRAGSearchRequest:
    """Request body for SurrealDB-backed RAG search."""

    query: str
    provider: str | None = None
    limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"query": self.query}
        if self.provider is not None:
            d["provider"] = self.provider
        if self.limit is not None:
            d["limit"] = self.limit
        return d


@dataclass
class SurrealRAGResult:
    """A single result from SurrealDB RAG search."""

    provider: str = ""
    title: str = ""
    heading: str = ""
    source_file: str = ""
    content: str = ""
    score: float = 0.0


@dataclass
class SurrealRAGSearchResponse:
    """Response from SurrealDB RAG search."""

    results: list[SurrealRAGResult] = field(default_factory=list)
    query: str = ""
    provider: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurrealRAGSearchResponse:
        results = [
            SurrealRAGResult(
                provider=r.get("provider", ""),
                title=r.get("title", ""),
                heading=r.get("heading", ""),
                source_file=r.get("source_file", ""),
                content=r.get("content", ""),
                score=r.get("score", 0.0),
            )
            for r in data.get("results", [])
        ]
        return cls(
            results=results,
            query=data.get("query", ""),
            provider=data.get("provider", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

@dataclass
class ModelInfo:
    """An available model."""

    id: str = ""
    provider: str = ""
    display_name: str = ""
    input_per_million: float = 0.0
    output_per_million: float = 0.0


@dataclass
class PricingInfo:
    """Pricing details for a model."""

    id: str = ""
    provider: str = ""
    display_name: str = ""
    input_per_million: float = 0.0
    output_per_million: float = 0.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TICKS_PER_USD: int = 10_000_000_000
