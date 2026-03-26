"""Extended types for Quantum AI API — voice library, finetunes, billing, job streaming,
search, agent/mission, compute, audio, video, documents, keys, credits, auth, mesh, realtime."""

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


# ---------------------------------------------------------------------------
# Agent / Mission
# ---------------------------------------------------------------------------

@dataclass
class AgentRequest:
    """Request body for an agent run."""

    task: str = ""
    session_id: str | None = None
    conductor_model: str | None = None
    workers: list[dict[str, Any]] | None = None
    max_steps: int | None = None
    system_prompt: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"task": self.task}
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.conductor_model is not None:
            d["conductor_model"] = self.conductor_model
        if self.workers is not None:
            d["workers"] = self.workers
        if self.max_steps is not None:
            d["max_steps"] = self.max_steps
        if self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        return d


@dataclass
class AgentStreamEvent:
    """A single event from an agent or mission SSE stream."""

    event_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentStreamEvent:
        event_type = data.pop("type", "")
        return cls(event_type=event_type, data=data)


@dataclass
class MissionWorker:
    """Describes a named worker for a mission."""

    model: str | None = None
    tier: str | None = None
    description: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.model is not None:
            d["model"] = self.model
        if self.tier is not None:
            d["tier"] = self.tier
        if self.description is not None:
            d["description"] = self.description
        return d


@dataclass
class MissionRequest:
    """Request body for a mission run."""

    goal: str = ""
    strategy: str | None = None
    conductor_model: str | None = None
    workers: dict[str, MissionWorker] | None = None
    max_steps: int | None = None
    system_prompt: str | None = None
    session_id: str | None = None
    auto_plan: bool | None = None
    context_config: dict[str, Any] | None = None
    worker_model: str | None = None
    deployment_id: str | None = None
    build_command: str | None = None
    workspace_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"goal": self.goal}
        if self.strategy is not None:
            d["strategy"] = self.strategy
        if self.conductor_model is not None:
            d["conductor_model"] = self.conductor_model
        if self.workers is not None:
            d["workers"] = {k: v.to_dict() for k, v in self.workers.items()}
        if self.max_steps is not None:
            d["max_steps"] = self.max_steps
        if self.system_prompt is not None:
            d["system_prompt"] = self.system_prompt
        if self.session_id is not None:
            d["session_id"] = self.session_id
        if self.auto_plan is not None:
            d["auto_plan"] = self.auto_plan
        if self.context_config is not None:
            d["context_config"] = self.context_config
        if self.worker_model is not None:
            d["worker_model"] = self.worker_model
        if self.deployment_id is not None:
            d["deployment_id"] = self.deployment_id
        if self.build_command is not None:
            d["build_command"] = self.build_command
        if self.workspace_path is not None:
            d["workspace_path"] = self.workspace_path
        return d


# ---------------------------------------------------------------------------
# Session Context
# ---------------------------------------------------------------------------

@dataclass
class SessionContext:
    """Context metadata returned with session responses."""

    turn_count: int = 0
    estimated_tokens: int = 0
    compacted: bool = False
    compaction_note: str | None = None


@dataclass
class ToolResult:
    """A tool result to feed back into a session."""

    tool_call_id: str = ""
    content: str = ""
    is_error: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "tool_call_id": self.tool_call_id,
            "content": self.content,
        }
        if self.is_error is not None:
            d["is_error"] = self.is_error
        return d


# ---------------------------------------------------------------------------
# Web Search
# ---------------------------------------------------------------------------

@dataclass
class WebSearchRequest:
    """Request body for Brave web search."""

    query: str = ""
    count: int | None = None
    offset: int | None = None
    country: str | None = None
    language: str | None = None
    freshness: str | None = None
    safesearch: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"query": self.query}
        if self.count is not None:
            d["count"] = self.count
        if self.offset is not None:
            d["offset"] = self.offset
        if self.country is not None:
            d["country"] = self.country
        if self.language is not None:
            d["language"] = self.language
        if self.freshness is not None:
            d["freshness"] = self.freshness
        if self.safesearch is not None:
            d["safesearch"] = self.safesearch
        return d


@dataclass
class WebResult:
    """A single web search result."""

    title: str = ""
    url: str = ""
    description: str = ""
    age: str | None = None
    favicon: str | None = None


@dataclass
class NewsResult:
    """A news search result."""

    title: str = ""
    url: str = ""
    description: str = ""
    age: str | None = None
    source: str | None = None


@dataclass
class VideoResult:
    """A video search result."""

    title: str = ""
    url: str = ""
    description: str = ""
    thumbnail: str | None = None
    age: str | None = None


@dataclass
class InfoboxResult:
    """An infobox (knowledge panel) result."""

    title: str = ""
    description: str = ""
    url: str | None = None


@dataclass
class DiscussionResult:
    """A discussion / forum result."""

    title: str = ""
    url: str = ""
    description: str = ""
    age: str | None = None
    forum: str | None = None


@dataclass
class WebSearchResponse:
    """Response from the web search endpoint."""

    query: str = ""
    web: list[WebResult] = field(default_factory=list)
    news: list[NewsResult] = field(default_factory=list)
    videos: list[VideoResult] = field(default_factory=list)
    infobox: list[InfoboxResult] = field(default_factory=list)
    discussions: list[DiscussionResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebSearchResponse:
        return cls(
            query=data.get("query", ""),
            web=[WebResult(**w) for w in data.get("web", [])],
            news=[NewsResult(**n) for n in data.get("news", [])],
            videos=[VideoResult(**v) for v in data.get("videos", [])],
            infobox=[InfoboxResult(**i) for i in data.get("infobox", [])],
            discussions=[DiscussionResult(**d) for d in data.get("discussions", [])],
        )


# ---------------------------------------------------------------------------
# Search Context
# ---------------------------------------------------------------------------

@dataclass
class SearchContextRequest:
    """Request body for search context (chunked page content)."""

    query: str = ""
    count: int | None = None
    country: str | None = None
    language: str | None = None
    freshness: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"query": self.query}
        if self.count is not None:
            d["count"] = self.count
        if self.country is not None:
            d["country"] = self.country
        if self.language is not None:
            d["language"] = self.language
        if self.freshness is not None:
            d["freshness"] = self.freshness
        return d


@dataclass
class SearchContextChunk:
    """A content chunk from search context."""

    content: str = ""
    url: str = ""
    title: str = ""
    score: float = 0.0
    content_type: str | None = None


@dataclass
class SearchContextSource:
    """A source reference from search context."""

    url: str = ""
    title: str = ""


@dataclass
class SearchContextResponse:
    """Response from the search context endpoint."""

    chunks: list[SearchContextChunk] = field(default_factory=list)
    sources: list[SearchContextSource] = field(default_factory=list)
    query: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchContextResponse:
        return cls(
            chunks=[SearchContextChunk(**c) for c in data.get("chunks", [])],
            sources=[SearchContextSource(**s) for s in data.get("sources", [])],
            query=data.get("query", ""),
        )


# ---------------------------------------------------------------------------
# Search Answer
# ---------------------------------------------------------------------------

@dataclass
class SearchAnswerMessage:
    """A chat message for the search answer endpoint."""

    role: str = ""
    content: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "content": self.content}


@dataclass
class SearchAnswerRequest:
    """Request body for search answer (AI answer grounded in search)."""

    messages: list[SearchAnswerMessage] = field(default_factory=list)
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"messages": [m.to_dict() for m in self.messages]}
        if self.model is not None:
            d["model"] = self.model
        return d


@dataclass
class SearchAnswerCitation:
    """A citation reference in a search answer."""

    url: str = ""
    title: str = ""
    snippet: str | None = None


@dataclass
class SearchAnswerChoice:
    """A choice in the search answer response."""

    index: int = 0
    message: SearchAnswerMessage | None = None
    finish_reason: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchAnswerChoice:
        msg_data = data.get("message")
        msg = SearchAnswerMessage(**msg_data) if msg_data else None
        return cls(
            index=data.get("index", 0),
            message=msg,
            finish_reason=data.get("finish_reason"),
        )


@dataclass
class SearchAnswerResponse:
    """Response from the search answer endpoint."""

    choices: list[SearchAnswerChoice] = field(default_factory=list)
    model: str = ""
    id: str = ""
    citations: list[SearchAnswerCitation] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SearchAnswerResponse:
        return cls(
            choices=[SearchAnswerChoice.from_dict(c) for c in data.get("choices", [])],
            model=data.get("model", ""),
            id=data.get("id", ""),
            citations=[SearchAnswerCitation(**c) for c in data.get("citations", [])],
        )


# ---------------------------------------------------------------------------
# Advanced Audio Types
# ---------------------------------------------------------------------------

@dataclass
class TtsRequest:
    """Request body for text-to-speech (Rust SDK name)."""

    model: str = ""
    text: str = ""
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
class TtsResponse:
    """Response from text-to-speech (Rust SDK name)."""

    audio_base64: str = ""
    format: str = ""
    size_bytes: int = 0
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""


@dataclass
class SttRequest:
    """Request body for speech-to-text (Rust SDK name)."""

    model: str = ""
    audio_base64: str = ""
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
class SttResponse:
    """Response from speech-to-text (Rust SDK name)."""

    text: str = ""
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""


@dataclass
class SoundEffectRequest:
    """Request body for sound effects generation."""

    prompt: str = ""
    duration_seconds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"prompt": self.prompt}
        if self.duration_seconds is not None:
            d["duration_seconds"] = self.duration_seconds
        return d


@dataclass
class SpeechToSpeechRequest:
    """Request body for speech-to-speech conversion."""

    audio_base64: str = ""
    model: str | None = None
    voice: str | None = None
    output_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"audio_base64": self.audio_base64}
        if self.model is not None:
            d["model"] = self.model
        if self.voice is not None:
            d["voice"] = self.voice
        if self.output_format is not None:
            d["format"] = self.output_format
        return d


@dataclass
class IsolateRequest:
    """Request body for voice isolation."""

    audio_base64: str = ""
    output_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"audio_base64": self.audio_base64}
        if self.output_format is not None:
            d["format"] = self.output_format
        return d


@dataclass
class RemixRequest:
    """Request body for voice remixing."""

    audio_base64: str = ""
    voice: str | None = None
    model: str | None = None
    output_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"audio_base64": self.audio_base64}
        if self.voice is not None:
            d["voice"] = self.voice
        if self.model is not None:
            d["model"] = self.model
        if self.output_format is not None:
            d["format"] = self.output_format
        return d


@dataclass
class DubRequest:
    """Request body for audio dubbing."""

    audio_base64: str = ""
    target_language: str = ""
    filename: str | None = None
    source_language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "audio_base64": self.audio_base64,
            "target_language": self.target_language,
        }
        if self.filename is not None:
            d["filename"] = self.filename
        if self.source_language is not None:
            d["source_language"] = self.source_language
        return d


@dataclass
class AlignRequest:
    """Request body for audio alignment / forced alignment."""

    audio_base64: str = ""
    text: str = ""
    language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"audio_base64": self.audio_base64, "text": self.text}
        if self.language is not None:
            d["language"] = self.language
        return d


@dataclass
class AlignmentSegment:
    """A single alignment segment."""

    text: str = ""
    start: float = 0.0
    end: float = 0.0


@dataclass
class AlignResponse:
    """Response from audio alignment."""

    segments: list[AlignmentSegment] = field(default_factory=list)
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AlignResponse:
        return cls(
            segments=[AlignmentSegment(**s) for s in data.get("segments", [])],
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class VoiceDesignRequest:
    """Request body for voice design."""

    description: str = ""
    text: str = ""
    output_format: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "voice_description": self.description,
            "sample_text": self.text,
        }
        if self.output_format is not None:
            d["format"] = self.output_format
        return d


@dataclass
class StarfishTTSRequest:
    """Request body for Starfish TTS."""

    text: str = ""
    voice: str | None = None
    output_format: str | None = None
    speed: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"text": self.text}
        if self.voice is not None:
            d["voice"] = self.voice
        if self.output_format is not None:
            d["format"] = self.output_format
        if self.speed is not None:
            d["speed"] = self.speed
        return d


@dataclass
class DialogueTurn:
    """A single dialogue turn."""

    speaker: str = ""
    text: str = ""
    voice: str | None = None


# ---------------------------------------------------------------------------
# Eleven Music (advanced)
# ---------------------------------------------------------------------------

@dataclass
class MusicSection:
    """A section within an Eleven Music generation request."""

    section_type: str = ""
    lyrics: str | None = None
    style: str | None = None
    style_exclude: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"section_type": self.section_type}
        if self.lyrics is not None:
            d["lyrics"] = self.lyrics
        if self.style is not None:
            d["style"] = self.style
        if self.style_exclude is not None:
            d["style_exclude"] = self.style_exclude
        return d


@dataclass
class ElevenMusicRequest:
    """Request body for advanced music generation (ElevenLabs)."""

    model: str = ""
    prompt: str = ""
    sections: list[MusicSection] | None = None
    duration_seconds: int | None = None
    language: str | None = None
    vocals: bool | None = None
    style: str | None = None
    style_exclude: str | None = None
    finetune_id: str | None = None
    edit_reference_id: str | None = None
    edit_instruction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"model": self.model, "prompt": self.prompt}
        if self.sections is not None:
            d["sections"] = [s.to_dict() for s in self.sections]
        if self.duration_seconds is not None:
            d["duration_seconds"] = self.duration_seconds
        if self.language is not None:
            d["language"] = self.language
        if self.vocals is not None:
            d["vocals"] = self.vocals
        if self.style is not None:
            d["style"] = self.style
        if self.style_exclude is not None:
            d["style_exclude"] = self.style_exclude
        if self.finetune_id is not None:
            d["finetune_id"] = self.finetune_id
        if self.edit_reference_id is not None:
            d["edit_reference_id"] = self.edit_reference_id
        if self.edit_instruction is not None:
            d["edit_instruction"] = self.edit_instruction
        return d


@dataclass
class ElevenMusicClip:
    """A single music clip from advanced generation."""

    base64: str = ""
    format: str = ""
    size: int = 0


@dataclass
class ElevenMusicResponse:
    """Response from advanced music generation."""

    clips: list[ElevenMusicClip] = field(default_factory=list)
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ElevenMusicResponse:
        return cls(
            clips=[ElevenMusicClip(**c) for c in data.get("clips", [])],
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class FinetuneInfo:
    """Info about a music finetune."""

    finetune_id: str = ""
    name: str = ""
    status: str = ""
    created_at: str | None = None


@dataclass
class ListFinetunesResponse:
    """Response from listing finetunes."""

    finetunes: list[FinetuneInfo] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ListFinetunesResponse:
        return cls(
            finetunes=[FinetuneInfo(**f) for f in data.get("finetunes", [])],
        )


# ---------------------------------------------------------------------------
# Voices (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class Voice:
    """A voice available for TTS."""

    voice_id: str = ""
    name: str = ""
    provider: str | None = None
    languages: list[str] | None = None
    gender: str | None = None
    is_cloned: bool | None = None
    preview_url: str | None = None


@dataclass
class VoicesResponse:
    """Response from listing voices."""

    voices: list[Voice] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VoicesResponse:
        return cls(voices=[Voice(**v) for v in data.get("voices", [])])


@dataclass
class CloneVoiceFile:
    """A file to include in a voice clone request."""

    filename: str = ""
    data: bytes = b""
    mime_type: str = ""


@dataclass
class CloneVoiceResponse:
    """Response from cloning a voice."""

    voice_id: str = ""
    name: str = ""
    status: str | None = None


@dataclass
class VoiceLibraryQuery:
    """Request parameters for browsing the voice library."""

    query: str | None = None
    page_size: int | None = None
    cursor: str | None = None
    gender: str | None = None
    language: str | None = None
    use_case: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.query is not None:
            d["query"] = self.query
        if self.page_size is not None:
            d["page_size"] = self.page_size
        if self.cursor is not None:
            d["cursor"] = self.cursor
        if self.gender is not None:
            d["gender"] = self.gender
        if self.language is not None:
            d["language"] = self.language
        if self.use_case is not None:
            d["use_case"] = self.use_case
        return d


# ---------------------------------------------------------------------------
# API Keys (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class CreateKeyRequest:
    """Request body for creating an API key."""

    name: str = ""
    endpoints: list[str] | None = None
    spend_cap_usd: float | None = None
    rate_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name}
        if self.endpoints is not None:
            d["endpoints"] = self.endpoints
        if self.spend_cap_usd is not None:
            d["spend_cap_usd"] = self.spend_cap_usd
        if self.rate_limit is not None:
            d["rate_limit"] = self.rate_limit
        return d


@dataclass
class KeyDetails:
    """Details about an API key."""

    id: str = ""
    name: str = ""
    key_prefix: str = ""
    scope: dict[str, Any] | None = None
    spent_ticks: int = 0
    revoked: bool = False
    created_at: str | None = None
    last_used: str | None = None


@dataclass
class CreateKeyResponse:
    """Response from creating an API key."""

    key: str = ""
    details: KeyDetails | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CreateKeyResponse:
        details_data = data.get("details")
        details = KeyDetails(**details_data) if details_data else None
        return cls(key=data.get("key", ""), details=details)


@dataclass
class ListKeysResponse:
    """Response from listing API keys."""

    keys: list[KeyDetails] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ListKeysResponse:
        return cls(keys=[KeyDetails(**k) for k in data.get("keys", [])])


@dataclass
class StatusResponse:
    """Generic status response."""

    status: str = ""
    message: str | None = None


# ---------------------------------------------------------------------------
# Compute (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class TemplatesResponse:
    """Response from listing compute templates."""

    templates: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProvisionRequest:
    """Request body for provisioning a compute instance."""

    template: str = ""
    zone: str | None = None
    spot: bool | None = None
    auto_teardown_minutes: int | None = None
    ssh_public_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"template": self.template}
        if self.zone is not None:
            d["zone"] = self.zone
        if self.spot is not None:
            d["spot"] = self.spot
        if self.auto_teardown_minutes is not None:
            d["auto_teardown_minutes"] = self.auto_teardown_minutes
        if self.ssh_public_key is not None:
            d["ssh_public_key"] = self.ssh_public_key
        return d


@dataclass
class ProvisionResponse:
    """Response from provisioning a compute instance."""

    instance_id: str = ""
    status: str = ""
    template: str | None = None
    zone: str | None = None
    ssh_address: str | None = None
    price_per_hour_usd: float | None = None


@dataclass
class InstancesResponse:
    """Response from listing compute instances."""

    instances: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class InstanceResponse:
    """Response from getting a single compute instance."""

    instance: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeleteResponse:
    """Response from deleting a compute instance."""

    status: str = ""
    instance_id: str | None = None


@dataclass
class SSHKeyRequest:
    """Request body for adding an SSH key to an instance."""

    ssh_public_key: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"ssh_public_key": self.ssh_public_key}


@dataclass
class BillingRequest:
    """Request for querying compute billing."""

    instance_id: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.instance_id is not None:
            d["instance_id"] = self.instance_id
        if self.start_date is not None:
            d["start_date"] = self.start_date
        if self.end_date is not None:
            d["end_date"] = self.end_date
        return d


# ---------------------------------------------------------------------------
# Video (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class JobResponse:
    """Response from async video job submission."""

    job_id: str = ""
    status: str = ""
    cost_ticks: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobResponse:
        known = {"job_id", "status", "cost_ticks"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            job_id=data.get("job_id", ""),
            status=data.get("status", ""),
            cost_ticks=data.get("cost_ticks", 0),
            extra=extra,
        )


@dataclass
class StudioClip:
    """A clip in a studio video."""

    avatar_id: str | None = None
    voice_id: str | None = None
    script: str | None = None
    background: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.avatar_id is not None:
            d["avatar_id"] = self.avatar_id
        if self.voice_id is not None:
            d["voice_id"] = self.voice_id
        if self.script is not None:
            d["script"] = self.script
        if self.background is not None:
            d["background"] = self.background
        return d


@dataclass
class StudioVideoRequest:
    """Request body for HeyGen studio video creation."""

    clips: list[StudioClip] = field(default_factory=list)
    title: str | None = None
    dimension: str | None = None
    aspect_ratio: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"clips": [c.to_dict() for c in self.clips]}
        if self.title is not None:
            d["title"] = self.title
        if self.dimension is not None:
            d["dimension"] = self.dimension
        if self.aspect_ratio is not None:
            d["aspect_ratio"] = self.aspect_ratio
        return d


@dataclass
class TranslateRequest:
    """Request body for video translation."""

    target_language: str = ""
    video_url: str | None = None
    video_base64: str | None = None
    source_language: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"target_language": self.target_language}
        if self.video_url is not None:
            d["video_url"] = self.video_url
        if self.video_base64 is not None:
            d["video_base64"] = self.video_base64
        if self.source_language is not None:
            d["source_language"] = self.source_language
        return d


@dataclass
class PhotoAvatarRequest:
    """Request body for creating a photo avatar video."""

    photo_base64: str = ""
    script: str = ""
    voice_id: str | None = None
    aspect_ratio: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "photo_base64": self.photo_base64,
            "script": self.script,
        }
        if self.voice_id is not None:
            d["voice_id"] = self.voice_id
        if self.aspect_ratio is not None:
            d["aspect_ratio"] = self.aspect_ratio
        return d


@dataclass
class DigitalTwinRequest:
    """Request body for digital twin video generation."""

    avatar_id: str = ""
    script: str = ""
    voice_id: str | None = None
    aspect_ratio: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "avatar_id": self.avatar_id,
            "script": self.script,
        }
        if self.voice_id is not None:
            d["voice_id"] = self.voice_id
        if self.aspect_ratio is not None:
            d["aspect_ratio"] = self.aspect_ratio
        return d


@dataclass
class Avatar:
    """A HeyGen avatar."""

    avatar_id: str = ""
    name: str | None = None
    gender: str | None = None
    preview_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class AvatarsResponse:
    """Response from listing HeyGen avatars."""

    avatars: list[Avatar] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AvatarsResponse:
        return cls(avatars=[Avatar(**a) for a in data.get("avatars", [])])


@dataclass
class VideoTemplate:
    """A HeyGen video template."""

    template_id: str = ""
    name: str | None = None
    preview_url: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class VideoTemplatesResponse:
    """Response from listing HeyGen video templates."""

    templates: list[VideoTemplate] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VideoTemplatesResponse:
        return cls(templates=[VideoTemplate(**t) for t in data.get("templates", [])])


@dataclass
class HeyGenVoicesResponse:
    """Response from listing HeyGen voices."""

    voices: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Documents (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class ChunkRequest:
    """Request body for document chunking."""

    file_base64: str = ""
    filename: str = ""
    max_chunk_tokens: int | None = None
    overlap_tokens: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "file_base64": self.file_base64,
            "filename": self.filename,
        }
        if self.max_chunk_tokens is not None:
            d["max_chunk_tokens"] = self.max_chunk_tokens
        if self.overlap_tokens is not None:
            d["overlap_tokens"] = self.overlap_tokens
        return d


@dataclass
class DocumentChunk:
    """A single document chunk."""

    index: int = 0
    text: str = ""
    token_count: int | None = None


@dataclass
class ChunkResponse:
    """Response from document chunking."""

    chunks: list[DocumentChunk] = field(default_factory=list)
    total_chunks: int | None = None
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ChunkResponse:
        return cls(
            chunks=[DocumentChunk(**c) for c in data.get("chunks", [])],
            total_chunks=data.get("total_chunks"),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class ProcessRequest:
    """Request body for document processing."""

    file_base64: str = ""
    filename: str = ""
    prompt: str | None = None
    model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "file_base64": self.file_base64,
            "filename": self.filename,
        }
        if self.prompt is not None:
            d["prompt"] = self.prompt
        if self.model is not None:
            d["model"] = self.model
        return d


@dataclass
class ProcessResponse:
    """Response from document processing."""

    content: str = ""
    model: str | None = None
    cost_ticks: int = 0
    request_id: str = ""


# ---------------------------------------------------------------------------
# Jobs (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class JobCreateRequest:
    """Request to create an async job."""

    job_type: str = ""
    params: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.job_type, "params": self.params}


@dataclass
class JobSummary:
    """Summary of a job in the list response."""

    job_id: str = ""
    status: str = ""
    job_type: str | None = None
    created_at: str | None = None
    completed_at: str | None = None
    cost_ticks: int = 0


@dataclass
class ListJobsResponse:
    """Response from listing jobs."""

    jobs: list[JobSummary] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ListJobsResponse:
        return cls(jobs=[JobSummary(**j) for j in data.get("jobs", [])])


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

@dataclass
class BatchJob:
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


# ---------------------------------------------------------------------------
# RAG (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class RagSearchRequest:
    """Request body for Vertex AI RAG search (Rust SDK name)."""

    query: str = ""
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
class RagResult:
    """A single result from RAG search (Rust SDK name)."""

    source_uri: str = ""
    source_name: str = ""
    text: str = ""
    score: float = 0.0
    distance: float = 0.0


@dataclass
class RagSearchResponse:
    """Response from RAG search (Rust SDK name)."""

    results: list[RagResult] = field(default_factory=list)
    query: str = ""
    corpora: list[str] | None = None
    cost_ticks: int = 0
    request_id: str = ""


@dataclass
class RagCorpus:
    """Describes an available RAG corpus (Rust SDK name)."""

    name: str = ""
    display_name: str = ""
    description: str = ""
    state: str = ""


@dataclass
class SurrealRagSearchRequest:
    """Request body for SurrealDB RAG search (Rust SDK name)."""

    query: str = ""
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
class SurrealRagResult:
    """A single result from SurrealDB RAG search (Rust SDK name)."""

    provider: str = ""
    title: str = ""
    heading: str = ""
    source_file: str = ""
    content: str = ""
    score: float = 0.0


@dataclass
class SurrealRagSearchResponse:
    """Response from SurrealDB RAG search (Rust SDK name)."""

    results: list[SurrealRagResult] = field(default_factory=list)
    query: str = ""
    provider: str | None = None
    cost_ticks: int = 0
    request_id: str = ""


@dataclass
class SurrealRagProvider:
    """A SurrealDB RAG provider (Rust SDK name)."""

    provider: str = ""
    chunk_count: int | None = None


@dataclass
class SurrealRagProvidersResponse:
    """Response from listing SurrealDB RAG providers (Rust SDK name)."""

    providers: list[SurrealRagProvider] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SurrealRagProvidersResponse:
        return cls(providers=[SurrealRagProvider(**p) for p in data.get("providers", [])])


# ---------------------------------------------------------------------------
# RAG Collections (user-scoped xAI proxy)
# ---------------------------------------------------------------------------

@dataclass
class Collection:
    id: str = ""
    name: str = ""
    description: str | None = None
    document_count: int | None = None
    owner: str | None = None
    provider: str | None = None
    created_at: str | None = None

@dataclass
class CollectionDocument:
    file_id: str = ""
    name: str = ""
    size_bytes: int | None = None
    content_type: str | None = None
    processing_status: str | None = None
    document_status: str | None = None
    indexed: bool | None = None
    created_at: str | None = None

@dataclass
class CollectionSearchResult:
    content: str = ""
    score: float | None = None
    file_id: str | None = None
    collection_id: str | None = None
    metadata: dict[str, Any] | None = None

@dataclass
class CollectionSearchRequest:
    query: str = ""
    collection_ids: list[str] = field(default_factory=list)
    mode: str | None = None
    max_results: int | None = None

@dataclass
class CollectionUploadResult:
    file_id: str = ""
    filename: str = ""
    bytes: int | None = None


# ---------------------------------------------------------------------------
# Error types (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class ApiError:
    """An error returned by the Quantum AI API."""

    status_code: int = 0
    code: str = ""
    message: str = ""
    request_id: str = ""

    def is_rate_limit(self) -> bool:
        return self.status_code == 429

    def is_auth(self) -> bool:
        return self.status_code in (401, 403)

    def is_not_found(self) -> bool:
        return self.status_code == 404


@dataclass
class Error:
    """Error type for SDK operations (mirrors Rust enum)."""

    api: ApiError | None = None
    message: str = ""


# ---------------------------------------------------------------------------
# Account / Credits (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class UsageQuery:
    """Usage query parameters."""

    limit: int | None = None
    start_after: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.limit is not None:
            d["limit"] = self.limit
        if self.start_after is not None:
            d["start_after"] = self.start_after
        return d


@dataclass
class CreditPurchaseRequest:
    """Request to purchase a credit pack."""

    pack_id: str = ""
    success_url: str | None = None
    cancel_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"pack_id": self.pack_id}
        if self.success_url is not None:
            d["success_url"] = self.success_url
        if self.cancel_url is not None:
            d["cancel_url"] = self.cancel_url
        return d


@dataclass
class DevProgramApplyRequest:
    """Request to apply for the developer program."""

    use_case: str = ""
    company: str | None = None
    expected_usd: float | None = None
    website: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"use_case": self.use_case}
        if self.company is not None:
            d["company"] = self.company
        if self.expected_usd is not None:
            d["expected_usd"] = self.expected_usd
        if self.website is not None:
            d["website"] = self.website
        return d


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

@dataclass
class AuthAppleRequest:
    """Request body for Apple Sign-In."""

    id_token: str = ""
    name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"id_token": self.id_token}
        if self.name is not None:
            d["name"] = self.name
        return d


# ---------------------------------------------------------------------------
# Response Meta
# ---------------------------------------------------------------------------

@dataclass
class ResponseMeta:
    """Metadata extracted from response headers."""

    cost_ticks: int = 0
    request_id: str = ""
    model: str = ""


# ---------------------------------------------------------------------------
# Realtime (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class RealtimeSession:
    """Response from the QAI realtime session endpoint."""

    ephemeral_token: str = ""
    url: str = ""
    signed_url: str = ""
    session_id: str = ""
    provider: str = ""

    def ws_url(self) -> str:
        """Get the WebSocket URL."""
        return self.signed_url if self.signed_url else self.url


@dataclass
class RealtimeEvent:
    """A parsed incoming event from the realtime API (Rust SDK name)."""

    event_type: str = ""
    data: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 3D Mesh pipeline (Rust SDK canonical names)
# ---------------------------------------------------------------------------

@dataclass
class RemeshRequest:
    """Request for a 3D remesh operation."""

    input_task_id: str | None = None
    model_url: str | None = None
    target_formats: list[str] | None = None
    topology: str | None = None
    target_polycount: int | None = None
    resize_height: float | None = None
    origin_at: str | None = None
    convert_format_only: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.input_task_id is not None:
            d["input_task_id"] = self.input_task_id
        if self.model_url is not None:
            d["model_url"] = self.model_url
        if self.target_formats is not None:
            d["target_formats"] = self.target_formats
        if self.topology is not None:
            d["topology"] = self.topology
        if self.target_polycount is not None:
            d["target_polycount"] = self.target_polycount
        if self.resize_height is not None:
            d["resize_height"] = self.resize_height
        if self.origin_at is not None:
            d["origin_at"] = self.origin_at
        if self.convert_format_only is not None:
            d["convert_format_only"] = self.convert_format_only
        return d


@dataclass
class ModelUrls:
    """URLs for each exported format in a remesh result."""

    glb: str = ""
    fbx: str = ""
    obj: str = ""
    usdz: str = ""
    stl: str = ""
    blend: str = ""


@dataclass
class RetextureRequest:
    """Request for AI retexturing of a 3D model."""

    prompt: str = ""
    input_task_id: str | None = None
    model_url: str | None = None
    enable_pbr: bool | None = None
    ai_model: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"prompt": self.prompt}
        if self.input_task_id is not None:
            d["input_task_id"] = self.input_task_id
        if self.model_url is not None:
            d["model_url"] = self.model_url
        if self.enable_pbr is not None:
            d["enable_pbr"] = self.enable_pbr
        if self.ai_model is not None:
            d["ai_model"] = self.ai_model
        return d


@dataclass
class RigRequest:
    """Request for auto-rigging a humanoid 3D model."""

    input_task_id: str | None = None
    model_url: str | None = None
    height_meters: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.input_task_id is not None:
            d["input_task_id"] = self.input_task_id
        if self.model_url is not None:
            d["model_url"] = self.model_url
        if self.height_meters is not None:
            d["height_meters"] = self.height_meters
        return d


@dataclass
class AnimationPostProcess:
    """Post-processing options for animation export."""

    operation_type: str = ""
    fps: int | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"operation_type": self.operation_type}
        if self.fps is not None:
            d["fps"] = self.fps
        return d


@dataclass
class AnimateRequest:
    """Request for applying an animation to a rigged character."""

    rig_task_id: str = ""
    action_id: int = 0
    post_process: AnimationPostProcess | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "rig_task_id": self.rig_task_id,
            "action_id": self.action_id,
        }
        if self.post_process is not None:
            d["post_process"] = self.post_process.to_dict()
        return d


# ---------------------------------------------------------------------------
# CreditTier extra field
# ---------------------------------------------------------------------------

@dataclass
class CreditTier:
    """A pricing tier (with extra field matching Rust SDK)."""

    name: str | None = None
    min_balance: int = 0
    discount_percent: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Citation (from chat.rs)
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """A citation reference in a chat response."""

    title: str = ""
    url: str = ""
    text: str = ""
    index: int = 0


# ---------------------------------------------------------------------------
# Additional Audio Types (cross-SDK parity with Rust audio.rs)
# ---------------------------------------------------------------------------

@dataclass
class AlignedWord:
    """A single word with timing information from forced alignment."""

    text: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    confidence: float = 0.0


@dataclass
class DialogueResponse:
    """Response from dialogue generation."""

    audio_base64: str = ""
    format: str = ""
    size_bytes: int = 0
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DialogueResponse:
        return cls(
            audio_base64=data.get("audio_base64", ""),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class DubResponse:
    """Response from dubbing."""

    dubbing_id: str = ""
    audio_base64: str = ""
    format: str = ""
    target_lang: str = ""
    status: str = ""
    processing_time_seconds: float = 0.0
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DubResponse:
        return cls(
            dubbing_id=data.get("dubbing_id", ""),
            audio_base64=data.get("audio_base64", ""),
            format=data.get("format", ""),
            target_lang=data.get("target_lang", ""),
            status=data.get("status", ""),
            processing_time_seconds=data.get("processing_time_seconds", 0.0),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class SpeechToSpeechResponse:
    """Response from speech-to-speech conversion."""

    audio_base64: str = ""
    format: str = ""
    size_bytes: int = 0
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SpeechToSpeechResponse:
        return cls(
            audio_base64=data.get("audio_base64", ""),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class IsolateVoiceResponse:
    """Response from voice isolation."""

    audio_base64: str = ""
    format: str = ""
    size_bytes: int = 0
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IsolateVoiceResponse:
        return cls(
            audio_base64=data.get("audio_base64", ""),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class RemixVoiceResponse:
    """Response from voice remixing."""

    audio_base64: str | None = None
    format: str = ""
    size_bytes: int = 0
    voice_id: str | None = None
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RemixVoiceResponse:
        return cls(
            audio_base64=data.get("audio_base64"),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            voice_id=data.get("voice_id"),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class VoicePreview:
    """A single voice preview from voice design."""

    generated_voice_id: str = ""
    audio_base64: str = ""
    format: str = ""


@dataclass
class StarfishTTSResponse:
    """Response from Starfish TTS."""

    audio_base64: str | None = None
    url: str | None = None
    format: str = ""
    size_bytes: int = 0
    duration: float = 0.0
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StarfishTTSResponse:
        return cls(
            audio_base64=data.get("audio_base64"),
            url=data.get("url"),
            format=data.get("format", ""),
            size_bytes=data.get("size_bytes", 0),
            duration=data.get("duration", 0.0),
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


# Aliases matching Rust IsolateVoiceRequest / RemixVoiceRequest
IsolateVoiceRequest = IsolateRequest
RemixVoiceRequest = RemixRequest


# ---------------------------------------------------------------------------
# Music Advanced Types (cross-SDK parity with Rust audio.rs)
# ---------------------------------------------------------------------------

@dataclass
class MusicAdvancedClip:
    """A single clip from advanced music generation."""

    base64: str = ""
    format: str = ""
    size: int = 0


@dataclass
class MusicAdvancedRequest:
    """Advanced music generation request."""

    prompt: str = ""
    duration_seconds: int | None = None
    model: str | None = None
    finetune_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"prompt": self.prompt}
        if self.duration_seconds is not None:
            d["duration_seconds"] = self.duration_seconds
        if self.model is not None:
            d["model"] = self.model
        if self.finetune_id is not None:
            d["finetune_id"] = self.finetune_id
        return d


@dataclass
class MusicAdvancedResponse:
    """Response from advanced music generation."""

    clips: list[MusicAdvancedClip] = field(default_factory=list)
    model: str = ""
    cost_ticks: int = 0
    request_id: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MusicAdvancedResponse:
        return cls(
            clips=[MusicAdvancedClip(**c) for c in data.get("clips", [])],
            model=data.get("model", ""),
            cost_ticks=data.get("cost_ticks", 0),
            request_id=data.get("request_id", ""),
        )


@dataclass
class MusicFinetuneCreateRequest:
    """Request to create a music finetune."""

    name: str = ""
    description: str | None = None
    samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"name": self.name, "samples": self.samples}
        if self.description is not None:
            d["description"] = self.description
        return d


# ---------------------------------------------------------------------------
# RAG Collection Wrapper Responses (cross-SDK parity with Rust rag.rs)
# ---------------------------------------------------------------------------

@dataclass
class CollectionDocumentsResponse:
    """Response from listing collection documents."""

    documents: list[CollectionDocument] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollectionDocumentsResponse:
        return cls(documents=[CollectionDocument(**d_) for d_ in data.get("documents", [])])


@dataclass
class CollectionSearchResponse:
    """Response from collection search."""

    results: list[CollectionSearchResult] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollectionSearchResponse:
        return cls(results=[CollectionSearchResult(**r) for r in data.get("results", [])])


@dataclass
class CollectionsListResponse:
    """Response from listing collections."""

    collections: list[Collection] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CollectionsListResponse:
        return cls(collections=[Collection(**c) for c in data.get("collections", [])])


@dataclass
class CreateCollectionRequest:
    """Request body for creating a collection."""

    name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name}


@dataclass
class DeleteCollectionResponse:
    """Response from deleting a collection."""

    message: str = ""


# ---------------------------------------------------------------------------
# Search — Discussion, Infobox (canonical Rust SDK names)
# ---------------------------------------------------------------------------

# Canonical Rust names point to existing types
Discussion = DiscussionResult
Infobox = InfoboxResult


# ---------------------------------------------------------------------------
# Models / RAG Wrapper Responses (cross-SDK parity)
# ---------------------------------------------------------------------------

@dataclass
class ModelsResponse:
    """Response from listing models."""

    models: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelsResponse:
        return cls(models=data.get("models", []))


@dataclass
class RagCorporaResponse:
    """Response from listing RAG corpora."""

    corpora: list = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RagCorporaResponse:
        return cls(corpora=data.get("corpora", []))


# Alias: Rust SDK uses SurrealRagProviderInfo as the canonical name
SurrealRagProviderInfo = SurrealRagProvider


# ---------------------------------------------------------------------------
# TTS/STT Aliases (long-form names matching Rust SDK)
# ---------------------------------------------------------------------------

TextToSpeechRequest = TtsRequest
TextToSpeechResponse = TtsResponse
SpeechToTextRequest = SttRequest
SpeechToTextResponse = SttResponse


# ---------------------------------------------------------------------------
# Phase 2: Remaining types for full Rust parity
# ---------------------------------------------------------------------------

@dataclass
class AddVoiceFromLibraryRequest:
    public_owner_id: str = ""
    voice_id: str = ""
    name: str = ""

@dataclass
class AgentEvent:
    type: str = ""
    delta: str = ""
    done: bool = False

@dataclass
class AgentWorkerConfig:
    model: str = ""
    tier: str = ""
    description: str = ""

MissionEvent = AgentEvent
MissionWorkerConfig = AgentWorkerConfig

@dataclass
class BasicAnimations:
    walking_glb: str = ""
    walking_fbx: str = ""
    running_glb: str = ""
    running_fbx: str = ""

@dataclass
class BatchSubmitRequest:
    jobs: list = field(default_factory=list)

@dataclass
class CloneVoiceRequest:
    name: str = ""
    description: str = ""
    audio_samples: list = field(default_factory=list)

@dataclass
class ComputeInstanceInfo:
    instance_id: str = ""
    template: str = ""
    status: str = ""
    zone: str = ""
    gpu_type: str = ""
    gpu_count: int = 0
    external_ip: str = ""
    hourly_usd: float = 0.0
    cost_usd: float = 0.0
    uptime_minutes: int = 0
    ssh_username: str = ""
    created_at: str = ""

@dataclass
class ContactResponse:
    status: str = ""
    message: str = ""

@dataclass
class ContextChunk:
    content: str = ""
    url: str = ""
    title: str = ""
    score: float = 0.0
    content_type: str = ""

@dataclass
class ContextMetadata:
    tools_cleared: bool = False

@dataclass
class ContextOptions:
    count: int = 0
    country: str = ""
    language: str = ""
    freshness: str = ""

Generate3DRequest = ImageRequest

@dataclass
class HeyGenAvatarsResponse:
    avatars: list = field(default_factory=list)
    request_id: str = ""

@dataclass
class HeyGenTemplatesResponse:
    templates: list = field(default_factory=list)
    request_id: str = ""

@dataclass
class JobAcceptedResponse:
    job_id: str = ""
    status: str = ""
    job_type: str = ""
    request_id: str = ""

@dataclass
class JobListEntry:
    job_id: str = ""
    status: str = ""
    job_type: str = ""
    cost_ticks: int = 0
    created_at: str = ""
    completed_at: str = ""
    request_id: str = ""

@dataclass
class LLMContextResponse:
    query: str = ""
    chunks: list = field(default_factory=list)
    sources: list = field(default_factory=list)

@dataclass
class PostProcess:
    operation_type: str = ""
    fps: int = 0

@dataclass
class SearchMessage:
    role: str = ""
    content: str = ""

@dataclass
class SearchOptions:
    count: int = 0
    offset: int = 0
    country: str = ""
    language: str = ""
    freshness: str = ""
    safe_search: str = ""

@dataclass
class SessionToolResult:
    tool_call_id: str = ""
    content: str = ""
    is_error: bool = False

@dataclass
class RealtimeSessionResponse:
    ephemeral_token: str = ""
    url: str = ""
    signed_url: str = ""
    session_id: str = ""
    provider: str = ""


# ---------------------------------------------------------------------------
# Scraper types
# ---------------------------------------------------------------------------

@dataclass
class ScrapeTarget:
    name: str = ""
    url: str = ""
    type: str = ""
    selector: str = ""
    content: str = ""
    notebook: str = ""
    recursive: bool = False
    max_pages: int = 0
    delay_ms: int = 0
    ingest: str = ""

@dataclass
class ScrapeRequest:
    targets: list = field(default_factory=list)

@dataclass
class ScrapeResponse:
    job_id: str = ""
    status: str = ""
    targets: int = 0
    request_id: str = ""

@dataclass
class ScreenshotURL:
    url: str = ""
    width: int = 0
    height: int = 0
    full_page: bool = False
    delay_ms: int = 0

@dataclass
class ScreenshotRequest:
    urls: list = field(default_factory=list)

@dataclass
class ScreenshotResult:
    url: str = ""
    base64: str = ""
    format: str = ""
    width: int = 0
    height: int = 0
    error: str = ""

@dataclass
class ScreenshotResponse:
    screenshots: list = field(default_factory=list)
    count: int = 0
