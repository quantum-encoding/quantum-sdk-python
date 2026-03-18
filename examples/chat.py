"""Example: basic chat with the Quantum AI API."""

import os
import sys

from quantum_sdk import Client, ChatMessage, APIError

def main() -> None:
    api_key = os.environ.get("QAI_API_KEY", "")
    if not api_key:
        print("Set QAI_API_KEY environment variable", file=sys.stderr)
        sys.exit(1)

    client = Client(api_key)

    # --- Non-streaming chat ---
    print("=== Non-streaming ===")
    resp = client.chat(
        model="claude-sonnet-4-6",
        messages=[
            ChatMessage.system("You are a helpful assistant."),
            ChatMessage.user("What is the capital of France?"),
        ],
    )
    print(f"Model: {resp.model}")
    print(f"Response: {resp.text()}")
    if resp.usage:
        print(f"Tokens: {resp.usage.input_tokens} in / {resp.usage.output_tokens} out")
    print(f"Cost ticks: {resp.cost_ticks}")
    print()

    # --- Streaming chat ---
    print("=== Streaming ===")
    try:
        for event in client.chat_stream(
            model="gpt-4.1-mini",
            messages=[ChatMessage.user("Count from 1 to 10.")],
        ):
            if event.delta:
                print(event.delta.text, end="", flush=True)
            if event.usage:
                print(f"\n[usage: {event.usage.input_tokens} in / {event.usage.output_tokens} out]")
            if event.error:
                print(f"\n[error: {event.error}]", file=sys.stderr)
            if event.done:
                print("\n[done]")
    except APIError as e:
        print(f"API error: {e}", file=sys.stderr)
        if e.is_rate_limit():
            print("Rate limited — try again later.", file=sys.stderr)

    client.close()


if __name__ == "__main__":
    main()
