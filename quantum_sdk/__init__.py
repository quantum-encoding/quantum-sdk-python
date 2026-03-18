"""Quantum AI Python SDK."""

from .client import Client, AsyncClient, DEFAULT_BASE_URL
from .errors import APIError, is_rate_limit_error, is_auth_error, is_not_found_error
from .types import (
    TICKS_PER_USD,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatTool,
    ChatUsage,
    ContentBlock,
    DocumentRequest,
    DocumentResponse,
    EmbedRequest,
    EmbedResponse,
    GeneratedImage,
    GeneratedVideo,
    ImageEditRequest,
    ImageEditResponse,
    ImageRequest,
    ImageResponse,
    ModelInfo,
    MusicClip,
    MusicRequest,
    MusicResponse,
    PricingInfo,
    RAGCorpus,
    RAGResult,
    RAGSearchRequest,
    RAGSearchResponse,
    STTRequest,
    STTResponse,
    StreamDelta,
    StreamEvent,
    StreamToolUse,
    SurrealRAGResult,
    SurrealRAGSearchRequest,
    SurrealRAGSearchResponse,
    TTSRequest,
    TTSResponse,
    VideoRequest,
    VideoResponse,
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
    # Chat
    "ChatMessage",
    "ChatRequest",
    "ChatResponse",
    "ChatTool",
    "ChatUsage",
    "ContentBlock",
    "StreamDelta",
    "StreamEvent",
    "StreamToolUse",
    # Image
    "GeneratedImage",
    "ImageEditRequest",
    "ImageEditResponse",
    "ImageRequest",
    "ImageResponse",
    # Video
    "GeneratedVideo",
    "VideoRequest",
    "VideoResponse",
    # Audio
    "MusicClip",
    "MusicRequest",
    "MusicResponse",
    "STTRequest",
    "STTResponse",
    "TTSRequest",
    "TTSResponse",
    # Embeddings
    "EmbedRequest",
    "EmbedResponse",
    # Documents
    "DocumentRequest",
    "DocumentResponse",
    # RAG
    "RAGCorpus",
    "RAGResult",
    "RAGSearchRequest",
    "RAGSearchResponse",
    "SurrealRAGResult",
    "SurrealRAGSearchRequest",
    "SurrealRAGSearchResponse",
    # Models
    "ModelInfo",
    "PricingInfo",
]

__version__ = "0.1.0"
