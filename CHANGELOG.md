# Changelog

Releases before 0.8.0 were tracked in git history only; see `git log`.

## Unreleased

**Breaking.** Every method left in this SDK calls a route the gateway serves.
The methods for routes the gateway removed, retired, or never had are gone
rather than left to fail at runtime.

### Removed
The gateway removed these routes on 2026-11-10 and now answers them with 404:
- User batch — `POST /qai/v1/batch`, `POST /qai/v1/batch/jsonl`,
  `GET /qai/v1/batch/jobs`, `GET /qai/v1/batch/jobs/{id}`:
  `batch_submit`, `batch_submit_jsonl`, `batch_jobs`, `batch_job` (both
  clients), the `quantum_sdk.batch` module, and `BatchJobInput`,
  `BatchSubmitResponse`, `BatchJsonlResponse`, `BatchJobInfo`,
  `BatchJobsResponse`, `BatchJob`, `BatchSubmitRequest`. HeyGen video batches
  (`video_batch_submit`, `video_batch_status`) are a different route and stay.
- Documents — `POST /qai/v1/documents/extract`, `/chunk`, `/process`:
  `extract_document`, `chunk_document`, `process_document` (both clients), the
  `quantum_sdk.documents` module, and `DocumentRequest`, `DocumentResponse`,
  `ChunkDocumentRequest`, `ChunkDocumentResponse`, `ProcessDocumentRequest`,
  `ProcessDocumentResponse`, `ChunkRequest`, `ChunkResponse`, `DocumentChunk`,
  `ProcessRequest`, `ProcessResponse`.
- Search answer — `POST /qai/v1/search/answer`: `search_answer` (both
  clients) and `SearchAnswerMessage`, `SearchAnswerRequest`,
  `SearchAnswerCitation`, `SearchAnswerChoice`, `SearchAnswerResponse`.
  `google_search` returns a grounded answer with its sources.
- Video studio — `POST /qai/v1/video/studio`: `video_studio` (both clients)
  and `VideoStudioRequest`, `StudioVideoRequest`, `StudioClip`.
- README rows for Documents, Scanner and Batch. This SDK never wrapped the
  scanner, conductor or model-deployment routes, which the gateway also
  removed.

These methods called paths that were already broken, either retired earlier
or never served:
- Scraper — `POST /qai/v1/scraper/scrape` and `/screenshot`, retired by the
  gateway on 2026-08-31: `scrape`, `screenshot` and `screenshot_job` (both
  clients). `screenshot_job` submitted a `"screenshot"` job, a type
  `POST /qai/v1/jobs` rejects with a 400. Gone with them: `ScrapeTarget`,
  `ScrapeRequest`, `ScrapeResponse`, `ScreenshotURL`, `ScreenshotRequest`,
  `ScreenshotResult`, `ScreenshotResponse`, `ScreenshotJobResponse`, and the
  README's Scraper row.
- Compute billing — `POST /qai/v1/compute/billing`, which the gateway has
  never served: `compute_billing` (both clients), `BillingRequest`,
  `BillingResponse`, `BillingEntry`.
- Collection documents — `GET /qai/v1/rag/collections/{id}/documents`, which
  the gateway has never served: `collections_documents` (both clients),
  `CollectionDocument`, `CollectionDocumentsResponse`.

### Fixed
- `collections_search` posted to `/qai/v1/rag/search/collections`, which the
  gateway does not serve. It now posts to `POST /qai/v1/rag/collections/search`.
  `CollectionSearchRequest` now carries the fields the handler reads: `query`,
  `collection_ids` (left out when empty, which searches every collection the
  caller can read) and `max_chunks` (gateway default 10). It replaces
  `max_results`. `mode` is removed because the gateway has no search modes.
  `CollectionSearchResult` now decodes the handler's result: `content`,
  `score`, `collection`, `collection_id`, `document_id`, `filename` and
  `is_shared`. It drops `file_id` and `metadata`, which the gateway never
  sent. It used to be built with `**result`, so any of the real keys raised
  TypeError. `CollectionSearchResponse` also keeps `query`,
  `collections_searched` and `request_id`.
- The 3D mesh helpers (`remesh`, `rig`, `animate`, `retexture`, sync and
  async) raised TypeError as soon as their job was created. They indexed the
  `JobCreateResponse` dataclass like a dict and passed `poll_job` an
  `interval_ms` it does not take. They now poll `job.job_id` every 5 s.

### Changed
- `video_photo_avatar` takes a `PhotoAvatarRequest` (`photo_base64`,
  `script`, optional `voice_id`, `aspect_ratio`), matching the Rust SDK. It
  used to send `{photo_url, script}`, a body the gateway does not accept: it
  reads only `photo_base64` + `script` and answers anything else with a 400.
  `VideoPhotoAvatarRequest`, the `photo_url` body, is removed.

## 0.9.0

The TTS request catches up with the gateway: a house voice you do not have to
name, and prose steering for Gemini.

### Added
- `TTSRequest.instructions` — style direction: tone, pace, accent, character.
  On Gemini it is prepended to the prompt and is the main way to steer a read,
  since Gemini exposes no knobs for any of it. On OpenAI only `gpt-4o-mini-tts`
  honours it; `tts-1`/`tts-1-hd` reject the field and the gateway drops it.
- `TTSRequest.language` — BCP-47 tag (`en-GB`, `es-ES`, `auto`). Gemini detects
  the language on its own; set this to pin the pronunciation or accent family.
  Also drives xAI pronunciation, where an English default sounds robotic on
  other languages.
- `TTSRequest.sample_rate` and `.bit_rate` — Hz and bits/sec, xAI only.
- `TTSRequest.voice_settings` and the new `TTSVoiceSettings` (`stability`,
  `similarity_boost`, `style`, `use_speaker_boost`). ElevenLabs only. Every
  field defaults to `None`: an absent knob leaves the provider default alone,
  and 0.0 stability is a real setting, so a zeroed object would silently
  retune the voice.
- `TTSRequest.speakers` and the new `TTSSpeaker` (`name`, `voice`) — Gemini
  two-voice dialogue. Each entry pairs a speaker label used in `text`
  ("Lacey: …") with the prebuilt voice that reads it. Exactly two; the gateway
  rejects any other count with a 400, and `voice` is then ignored.
- `VoiceInfo.provider`, `.model`, `.is_cloned` and `.description`.
  `GET /qai/v1/voices` has always sent these and this SDK dropped them, so a
  caller could not tell which provider served a voice, nor which model to pass
  back for it, without hardcoding the mapping.
- `TTSSpeaker` and `TTSVoiceSettings` are exported from the package root.

### Changed
- `TTSRequest.model` now defaults to `""` and is omitted from the body when
  empty, so `TTSRequest(text="…")` is a complete request and the gateway
  supplies its house default — `gemini-3.1-flash-tts-preview` with the
  `Laomedeia` voice. Previously an unset model was sent as `"model": ""`,
  pinning the request to a model that does not exist.

  `text` also gains a default, so that `model` can keep its position as the
  first field: existing positional calls, `TTSRequest("model", "text")`, are
  unaffected.

### Fixed
- The README's Text-to-Speech example did not run: it called `speak` with a
  bare string plus keyword arguments (the real signature takes a `TTSRequest`)
  and printed `audio.audio_url`, an attribute `TTSResponse` has never had. The
  audio arrives inline as `audio_base64`; there is no URL to fetch.

### Docs
- README gains "Steering a Gemini voice", summarising the gateway's
  `docs/TTS_GUIDE.md`: `instructions` for tone/accent/pace, the inline audio
  tags (`[whispers]`, `[excited]`, …) that go inside `text`, two-speaker
  dialogue, the 30 Gemini prebuilt voices, and which fields are xAI- or
  ElevenLabs-only.

Additive against an older gateway: the new fields are simply absent.

## 0.8.0

The reasoning state a tool loop has to hand back, and the cache key that keeps a
conversation on one shard.

### Added
- `ChatRequest.prompt_cache_key` — any stable string the client keeps per
  conversation. The gateway hashes it with the caller's identity and forwards it
  as OpenAI/xAI `prompt_cache_key` (or `x-grok-conv-id` on the xAI
  chat-completions lane), so every turn of one conversation lands on the same
  warm provider cache shard. `None` = derived from the caller's identity alone,
  which puts all of that user's conversations on one shard. Generate one per
  conversation object and reuse it on every turn.

  `POST /qai/v1/chat` only: the session endpoint derives its key from the
  session ID and ignores a client-supplied one.

- `ContentBlock.reasoning` and `ContentBlock.minted_by`, on blocks of the new
  type `"reasoning"`, carried through both `to_dict()` and `from_dict()`. This
  is the provider's own reasoning item, verbatim and opaque. It arrives
  interleaved with the `tool_use` blocks and must be echoed back unchanged,
  **in the position it arrived in**, on the next turn's assistant message: its
  place among the tool calls is how the provider learns where the reasoning
  sat, and replaying it behind the call it reasoned about is a different
  conversation the provider rejects. Dropping it re-bills the reasoning tokens
  on every round of a tool loop.

  `minted_by` names the model that produced the block; reasoning state is bound
  to its model and is never replayed to a different one.

  Distinct from the existing `"thinking"` block, which is the human-readable
  summary of the same turn. One is for the reader, one is for the wire.

- `StreamEvent.thought_signature`, carried by the new `thought_signature` SSE
  event the gateway sends just before `done` on a Gemini 3 stream that ended in
  text, and by the atomic `tool_use` event — which a streaming tool loop
  previously had no way to read.

### Changed
- `ContentBlock.thought_signature` is documented on **text** blocks as well as
  `tool_use` blocks: Gemini 3 signs a turn that ends in text. The field already
  accepted it — this states the contract, it is not a shape change.
- `provider_options` is documented as an open map, one JSON object per provider
  key plus the flat `region` entry. Newly documented:
  `provider_options["openai"]["reasoning_summary"]`
  (`auto` | `concise` | `detailed` | `none`), `["reasoning_mode"]`
  (`standard` | `pro`), `["verbosity"]` (`low` | `medium` | `high`),
  `["text_format"]` (`text` | `json_object`), and
  `provider_options["xai"]["native_files"]` (bool). The annotation was already
  `dict[str, Any]`, so unnamed keys always rode through.
- `reasoning_effort` documents `max`, which the gateway has always validated on
  every lane.

Additive against an older gateway: the new fields are simply absent.
