"""Quantum AI Python SDK."""

from .client import Client, AsyncClient, DEFAULT_BASE_URL
from .errors import APIError, is_rate_limit_error, is_auth_error, is_not_found_error
from .types import (
    TICKS_PER_USD,
    # Agent
    AgentRunRequest,
    AgentWorker,
    MissionRunRequest,
    # Alignment / Advanced Audio
    AlignmentResponse,
    AlignmentWord,
    AudioResponse,
    DialogueRequest,
    DialogueVoice,
    VoiceDesignResponse,
    # API Keys
    APIKeyCreateRequest,
    APIKeyCreateResponse,
    APIKeyInfo,
    APIKeyListResponse,
    # Account
    BalanceResponse,
    # Chat
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatTool,
    ChatUsage,
    ContentBlock,
    # Chunk / Process Documents
    ChunkDocumentRequest,
    ChunkDocumentResponse,
    ProcessDocumentRequest,
    ProcessDocumentResponse,
    # Compute
    ComputeInstance,
    ComputeProvisionRequest,
    ComputeProvisionResponse,
    ComputeTemplate,
    # Contact
    ContactRequest,
    ContactResponse,
    # Documents
    DocumentRequest,
    DocumentResponse,
    # Embeddings
    EmbedRequest,
    EmbedResponse,
    # Image
    GeneratedImage,
    ImageEditRequest,
    ImageEditResponse,
    ImageRequest,
    ImageResponse,
    # Jobs
    JobCreateResponse,
    JobListResponse,
    JobStatusResponse,
    # Models
    ModelInfo,
    PricingInfo,
    # Music / Audio
    MusicClip,
    MusicRequest,
    MusicResponse,
    SoundEffectResponse,
    STTRequest,
    STTResponse,
    TTSRequest,
    TTSResponse,
    # Pricing
    PricingEntry,
    PricingResponse,
    # RAG
    RAGCorpus,
    RAGResult,
    RAGSearchRequest,
    RAGSearchResponse,
    SurrealRAGProvider,
    SurrealRAGProvidersResponse,
    SurrealRAGResult,
    SurrealRAGSearchRequest,
    SurrealRAGSearchResponse,
    # Session Chat
    ContextConfig,
    SessionChatRequest,
    SessionChatResponse,
    # Streaming
    StreamDelta,
    StreamEvent,
    StreamToolUse,
    # Usage
    UsageEntry,
    UsageResponse,
    UsageSummaryMonth,
    UsageSummaryResponse,
    # Video
    GeneratedVideo,
    HeyGenAvatar,
    HeyGenTemplate,
    HeyGenVoice,
    VideoDigitalTwinRequest,
    VideoPhotoAvatarRequest,
    VideoRequest,
    VideoResponse,
    VideoStudioRequest,
    VideoTranslateRequest,
    # Voices
    VoiceCloneResponse,
    VoiceInfo,
    VoiceListResponse,
)

__all__ = [
    # Clients
    "Client",
    "AsyncClient",
    "DEFAULT_BASE_URL",
    # Errors
    "APIError",
    "is_rate_limit_error",
    "is_auth_error",
    "is_not_found_error",
    # Constants
    "TICKS_PER_USD",
    # Agent
    "AgentRunRequest",
    "AgentWorker",
    "MissionRunRequest",
    # Advanced Audio
    "AlignmentResponse",
    "AlignmentWord",
    "AudioResponse",
    "DialogueRequest",
    "DialogueVoice",
    "VoiceDesignResponse",
    # API Keys
    "APIKeyCreateRequest",
    "APIKeyCreateResponse",
    "APIKeyInfo",
    "APIKeyListResponse",
    # Chat
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatTool",
    "ChatUsage",
    "ContentBlock",
    # Chunk / Process Documents
    "ChunkDocumentRequest",
    "ChunkDocumentResponse",
    "ProcessDocumentRequest",
    "ProcessDocumentResponse",
    # Compute
    "ComputeInstance",
    "ComputeProvisionRequest",
    "ComputeProvisionResponse",
    "ComputeTemplate",
    # Contact
    "ContactRequest",
    "ContactResponse",
    # Documents
    "DocumentRequest",
    "DocumentResponse",
    # Embeddings
    "EmbedRequest",
    "EmbedResponse",
    # Image
    "GeneratedImage",
    "ImageEditRequest",
    "ImageEditResponse",
    "ImageRequest",
    "ImageResponse",
    # Jobs
    "JobCreateResponse",
    "JobListResponse",
    "JobStatusResponse",
    # Models
    "ModelInfo",
    "PricingInfo",
    # Audio
    "MusicClip",
    "MusicRequest",
    "MusicResponse",
    "SoundEffectResponse",
    "STTRequest",
    "STTResponse",
    "TTSRequest",
    "TTSResponse",
    # Pricing
    "PricingEntry",
    "PricingResponse",
    # RAG
    "RAGCorpus",
    "RAGResult",
    "RAGSearchRequest",
    "RAGSearchResponse",
    "SurrealRAGProvider",
    "SurrealRAGProvidersResponse",
    "SurrealRAGResult",
    "SurrealRAGSearchRequest",
    "SurrealRAGSearchResponse",
    # Session Chat
    "ContextConfig",
    "SessionChatRequest",
    "SessionChatResponse",
    # Streaming
    "StreamDelta",
    "StreamEvent",
    "StreamToolUse",
    # Account / Usage
    "BalanceResponse",
    "UsageEntry",
    "UsageResponse",
    "UsageSummaryMonth",
    "UsageSummaryResponse",
    # Video
    "GeneratedVideo",
    "HeyGenAvatar",
    "HeyGenTemplate",
    "HeyGenVoice",
    "VideoDigitalTwinRequest",
    "VideoPhotoAvatarRequest",
    "VideoRequest",
    "VideoResponse",
    "VideoStudioRequest",
    "VideoTranslateRequest",
    # Voices
    "VoiceCloneResponse",
    "VoiceInfo",
    "VoiceListResponse",
]

__version__ = "0.2.0"
