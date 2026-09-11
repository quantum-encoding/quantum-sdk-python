# Changelog

Releases before 0.8.0 were tracked in git history only; see `git log`.

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
