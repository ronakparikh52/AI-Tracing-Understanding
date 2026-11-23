"""Mini-Bank agent — Step 1: a single raw API call. No tools, no loop.

Goal: see exactly what goes over the wire and what comes back.
Run (from the project root):  python step1/agent.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find db.py / common.py
from common import save_output

save_output(__file__)  # everything printed also goes to step1/agent.txt

import anthropic

# The client holds credentials + HTTP settings. With no arguments it looks for
# ANTHROPIC_API_KEY, then ANTHROPIC_AUTH_TOKEN, then your `ant auth login` profile.
client = anthropic.Anthropic()

# Haiku 5.5: ~40x cheaper than Opus 5.5 — ideal for running experiments many times.
# Swap to "claude-opus-5-5" later to compare behavior (e.g. drift, tool choices).
MODEL = "claude-haiku-5-5"

# The system prompt: instructions that sit "above" the conversation.
# It is part of every request — the model has no memory between calls.
SYSTEM = "You are a support agent for Mini-Bank. Be concise and never guess account data."

# The conversation so far. Right now it's one user turn. In the agent loop
# (Step 2+) this list grows and gets re-sent IN FULL on every call.
messages = [
    {"role": "user", "content": "What's Alice's balance?"},
]

# One HTTP POST to /v1/messages. Everything the model knows is in these args.
response = client.messages.create(
    model=MODEL,
    max_tokens=16000,  # hard cap on output tokens; hitting it cuts the reply off
    system=SYSTEM,
    messages=messages,
)

# --- Inspect the response ---------------------------------------------------

# content is a LIST of typed blocks, not a string. Expect a "thinking" block
# (Haiku 5.5 thinks by default; its text is hidden by default) and then "text".
print("=== content blocks ===")
for i, block in enumerate(response.content):
    print(f"[{i}] type={block.type}")
    if block.type == "text":
        print(f"    {block.text}")

# Why the model stopped: "end_turn" = finished naturally, "max_tokens" = cut off,
# "tool_use" = wants you to run a tool (Step 2), "refusal" = declined.
print("\n=== stop_reason ===")
print(response.stop_reason)

# Token accounting — what you're billed for, and the raw data for the cache
# experiment later:
#   input_tokens                 uncached input, full price
#   cache_creation_input_tokens  input written to the prefix cache (~1.25x price)
#   cache_read_input_tokens      input served from cache (~0.1x price)
#   output_tokens                everything generated, INCLUDING hidden thinking
print("\n=== usage ===")
u = response.usage
print(f"input_tokens:                {u.input_tokens}")
print(f"cache_creation_input_tokens: {u.cache_creation_input_tokens}")
print(f"cache_read_input_tokens:     {u.cache_read_input_tokens}")
print(f"output_tokens:               {u.output_tokens}")
