# quantum-sdk-python

## SDK Parity Check

This SDK must stay in sync with the Rust reference SDK. Use `sdk-graph` to check parity:

```bash
# Scan this SDK (run after making changes)
sdk-graph scan --sdk python --dir ~/work/tauri_apps/qe-sdk-collection/python_projects/quantum-sdk/quantum_sdk

# Scan Rust reference (if not recently scanned)
sdk-graph scan --sdk rust --dir ~/work/tauri_apps/qe-sdk-collection/rust_projects/quantum-sdk/src

# Show what this SDK is missing vs Rust
sdk-graph diff --base rust --target python

# Show overall stats
sdk-graph stats
```

Binary: `~/go/bin/sdk-graph` (in PATH)
Graph file: `~/work/go_programs/quantum-ai/sdk-graph.json` (shared across all SDKs)

## Workflow

1. Before starting work: run `sdk-graph diff --base rust --target python` to see current gaps
2. After adding types/fields: rescan with `sdk-graph scan --sdk python --dir ~/work/tauri_apps/qe-sdk-collection/python_projects/quantum-sdk/quantum_sdk`
3. Verify gap reduced: run diff again
4. Goal: zero missing types and fields vs Rust

## Reference Implementation

The Rust SDK is the source of truth: `~/work/tauri_apps/qe-sdk-collection/rust_projects/quantum-sdk/src/`

When adding missing types, follow the Rust SDK's field names and JSON serialization tags exactly. Map types idiomatically:
- Rust `Option<T>` -> Python `Optional[T] = None` or `T | None = None`
- Rust `Vec<T>` -> Python `list[T]`
- Rust `String` -> Python `str`
- Rust serde rename -> Python dataclass field names (snake_case matches)

## API Server

Backend: https://api.quantumencoding.ai
Repo: ~/work/go_programs/quantum-ai
