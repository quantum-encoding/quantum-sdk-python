# quantum-sdk

Python client SDK for the [Quantum AI API](https://api.quantumencoding.ai).

```bash
pip install quantum-sdk
```

## Quick Start

```python
from quantum_sdk import Client

client = Client("qai_k_your_key_here")
response = client.chat("gemini-2.5-flash", "Hello! What is quantum computing?")
print(response.text)
```

## Features

- 110+ endpoints across 10 AI providers and 45+ models
- Sync and async clients (`Client` and `AsyncClient`)
- Dataclass responses with full type hints
- Streaming via generators and async generators
- Agent orchestration with SSE event streams
- GPU/CPU compute rental (requires per-account admin approval)
- Batch processing (50% discount)
- Python 3.10+

## Examples

### Chat Completion

```python
from quantum_sdk import Client

client = Client("qai_k_your_key_here")

response = client.chat(
    model="claude-sonnet-4-6",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Explain decorators in Python"},
    ],
    temperature=0.7,
    max_tokens=1000,
)
print(response.text)
```

### Streaming

```python
for event in client.chat_stream(
    model="claude-sonnet-4-6",
    messages=[{"role": "user", "content": "Write a haiku about Python"}],
):
    if event.delta_text:
        print(event.delta_text, end="", flush=True)
```

### Async Client

```python
from quantum_sdk import AsyncClient

async def main():
    client = AsyncClient("qai_k_your_key_here")
    response = await client.chat("gemini-2.5-flash", "Hello!")
    print(response.text)

    async for event in client.chat_stream(
        model="claude-sonnet-4-6",
        messages=[{"role": "user", "content": "Write a poem"}],
    ):
        if event.delta_text:
            print(event.delta_text, end="", flush=True)
```

### Reasoning state across a tool loop

Reasoning models on the OpenAI and xAI lanes mint a `reasoning` content block
alongside their `tool_use` blocks. It is the provider's own state, opaque, and
it must go back **unchanged and in the same position** on the next turn's
assistant message — its place among the tool calls is how the provider learns
where the reasoning sat. Drop it and the reasoning tokens are re-billed on
every round of the loop.

The simplest correct thing is to hand the whole `content` list back:

```python
messages = [ChatMessage(role="user", content="What is the weather in Oslo?")]

resp = client.chat(ChatRequest(
    model="gpt-5.6",
    messages=messages,
    # One key per conversation, reused on every turn, so all of them land on
    # the same warm provider cache shard.
    prompt_cache_key="conv-7f3a",
))

# Verbatim, in order: reasoning blocks, tool_use blocks, text blocks.
messages.append(ChatMessage(role="assistant", content_blocks=resp.content))
# ... then append one tool-result message per tool_use block and loop.
```

`ContentBlock.reasoning` holds the provider's item as it arrived — never
inspect or rebuild it. A `thinking` block is the human-readable summary of the
same turn: render that one, replay this one.

On Gemini 3 the equivalent state is `ContentBlock.thought_signature`, and it
now rides the **text** block of a turn that ended in text as well as the
`tool_use` blocks. Streaming delivers it as a `thought_signature` event just
before `done`, on `StreamEvent.thought_signature`.

### Provider options

`provider_options` is an open map, so a key the gateway documents but this SDK
version does not name still rides through — as does the flat `region` entry:

```python
ChatRequest(
    model="gpt-5.6",
    messages=messages,
    provider_options={
        "openai": {
            "reasoning_summary": "detailed",   # auto | concise | detailed | none
            "reasoning_mode": "pro",           # standard | pro
            "verbosity": "low",                # low | medium | high
            "text_format": "json_object",      # text | json_object
        },
        "xai": {"native_files": True},
        "region": "europe",                    # americas | europe | asia
    },
)
```

`reasoning_effort` accepts `none`, `low`, `medium`, `high`, `xhigh` and `max` on
every lane; each adapter folds a tier its model lacks onto the nearest one.

### Image Generation

```python
images = client.generate_image("grok-imagine-image", "A cosmic duck in space")
for image in images.images:
    print(image.url or "base64")
```

### Text-to-Speech

```python
audio = client.speak("Welcome to Quantum AI!", voice="alloy", format="mp3")
print(audio.audio_url)
```

### Web Search

```python
results = client.web_search("latest Python releases 2026")
for result in results.results:
    print(f"{result.title}: {result.url}")
```

### Agent Orchestration

```python
for event in client.agent_run("Research quantum computing breakthroughs"):
    if event.type == "content_delta":
        print(event.content or "", end="")
    elif event.type == "done":
        print("\n--- Done ---")
```

## All Endpoints

| Category | Endpoints | Description |
|----------|-----------|-------------|
| Chat | 2 | Text generation + session chat |
| Agent | 2 | Multi-step orchestration + missions |
| Images | 2 | Generation + editing |
| Video | 7 | Generation, studio, translation, avatars |
| Audio | 13 | TTS, STT, music, dialogue, dubbing, voice design |
| Voices | 5 | Clone, list, delete, library, design |
| Embeddings | 1 | Text embeddings |
| RAG | 4 | Vertex AI + SurrealDB search |
| Documents | 3 | Extract, chunk, process |
| Search | 3 | Web search, context, answers |
| Scanner | 11 | Code scanning, type queries, diffs |
| Scraper | 2 | Doc scraping + screenshots |
| Jobs | 3 | Async job management |
| Compute | 7 | GPU/CPU rental (admin-approved accounts only) |
| Keys | 3 | API key management |
| Account | 3 | Balance, usage, summary |
| Credits | 6 | Packs, tiers, lifetime, purchase |
| Batch | 4 | 50% discount batch processing |
| Realtime | 3 | Voice sessions |
| Models | 2 | Model list + pricing |

## Authentication

Pass your API key when creating the client:

```python
client = Client("qai_k_your_key_here")
```

The SDK sends it as the `X-API-Key` header. Both `qai_...` (primary) and `qai_k_...` (scoped) keys are supported. You can also use `Authorization: Bearer <key>`.

Get your API key at [cosmicduck.dev](https://cosmicduck.dev).

## Pricing

See [api.quantumencoding.ai/pricing](https://api.quantumencoding.ai/pricing) for current rates.

The **Lifetime tier** offers 0% margin at-cost pricing via a one-time payment.

## Other SDKs

All SDKs are at v0.4.0 with type parity verified by scanner.

| Language | Package | Install |
|----------|---------|---------|
| Rust | quantum-sdk | `cargo add quantum-sdk` |
| Go | quantum-sdk | `go get github.com/quantum-encoding/quantum-sdk` |
| TypeScript | @quantum-encoding/quantum-sdk | `npm i @quantum-encoding/quantum-sdk` |
| **Python** | quantum-sdk | `pip install quantum-sdk` |
| Swift | QuantumSDK | Swift Package Manager |
| Kotlin | quantum-sdk | Gradle dependency |

MCP server: `npx @quantum-encoding/ai-conductor-mcp`

## API Reference

- Interactive docs: [api.quantumencoding.ai/docs](https://api.quantumencoding.ai/docs)
- OpenAPI spec: [api.quantumencoding.ai/openapi.yaml](https://api.quantumencoding.ai/openapi.yaml)
- LLM context: [api.quantumencoding.ai/llms.txt](https://api.quantumencoding.ai/llms.txt)

## License

MIT
