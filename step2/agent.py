"""Mini-Bank agent — Step 2: one tool + the agent loop.

An "agent" = this loop:
    1. send the whole conversation + tool definitions to the model
    2. if the model asks for a tool (stop_reason == "tool_use"), WE run it
    3. append the model's turn and our tool results to the conversation
    4. repeat until the model stops asking for tools

Run (from the project root):  python step2/agent.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find db.py / common.py
from common import save_output
from db import query

import anthropic

client = anthropic.Anthropic()

MODEL = "claude-haiku-5-5"

# Print the exact JSON body sent to /v1/messages and the full JSON response,
# every turn. Set False for just the short summary lines.
SHOW_RAW_JSON = True

SYSTEM = (
    "You are a support agent for Mini-Bank. Use your tools to look up data; "
    "never guess account data. Amounts from tools are in integer cents - "
    "show dollars to the customer."
)

# --- Tool definitions (what the MODEL sees) ---------------------------------
# The model never sees your Python. It only sees this JSON: a name, a
# description (its only guide for WHEN to call it), and a JSON Schema for the
# arguments. These bytes are part of every request - and of the cache prefix.
TOOLS = [
    {
        "name": "get_account",
        "description": (
            "Look up Mini-Bank accounts by owner name (case-insensitive, partial "
            "match). Returns id, owner_name and balance_cents for each match, or "
            "an error if nothing matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "owner_name": {
                    "type": "string",
                    "description": "Full or partial name, e.g. 'Alice' or 'Alice Chen'.",
                }
            },
            "required": ["owner_name"],
            "additionalProperties": False,
        },
        "strict": True,  # API guarantees the model's arguments match the schema
    }
]


# --- Tool implementations (what YOUR CODE runs) -----------------------------
def get_account(owner_name: str) -> dict:
    rows = query(
        "SELECT id, owner_name, balance_cents FROM accounts WHERE owner_name ILIKE %s",
        (f"%{owner_name}%",),  # parameterized: the model's text never becomes SQL
    )
    if not rows:
        return {"error": f"No account found matching '{owner_name}'"}
    return {"accounts": rows}


# Dispatch table: tool name (from the model) -> Python function.
TOOL_FUNCTIONS = {"get_account": get_account}


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """Run one tool call. Returns (result_as_text, is_error).

    Errors are returned to the model as data, not raised - so the model can
    see what went wrong and recover (retry, ask the user, etc.).
    """
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"}), True
    try:
        return json.dumps(fn(**tool_input), default=str), False
    except Exception as e:
        return json.dumps({"error": str(e)}), True


# --- Raw JSON printing --------------------------------------------------------
def _jsonable(obj):
    """json.dumps fallback: SDK objects (e.g. the assistant's content blocks we
    appended to `messages`) are Pydantic models - convert them to plain dicts."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    raise TypeError(f"Can't serialize {type(obj)}")


def print_json(label: str, data) -> None:
    print(f"\n  {label}")
    print(json.dumps(data, indent=2, default=_jsonable))


# --- The agent loop ---------------------------------------------------------
def run_agent(user_message: str, max_turns: int = 10) -> list:
    messages = [{"role": "user", "content": user_message}]

    for turn in range(1, max_turns + 1):
        # Stateless call: the FULL history goes up every time.
        # Built as a dict so we can print exactly what gets sent.
        request = {
            "model": MODEL,
            "max_tokens": 16000,
            "system": SYSTEM,
            "tools": TOOLS,
            "messages": messages,
        }
        if SHOW_RAW_JSON:
            print_json(f">>> REQUEST (turn {turn}) -> POST /v1/messages", request)

        response = client.messages.create(**request)

        if SHOW_RAW_JSON:
            print_json(f"<<< RESPONSE (turn {turn})", response.model_dump(exclude_none=True))

        u = response.usage
        print(f"\n--- turn {turn} | stop_reason={response.stop_reason} "
              f"| in={u.input_tokens} out={u.output_tokens}")
        for block in response.content:
            if block.type == "text":
                print(f"  [text] {block.text}")
            elif block.type == "tool_use":
                print(f"  [tool_use] {block.name}({json.dumps(block.input)})  id={block.id}")
            else:
                print(f"  [{block.type}]")  # e.g. thinking

        # Append the model's turn EXACTLY as returned (all blocks, incl. any
        # thinking). Rewriting or dropping blocks breaks the next request.
        messages.append({"role": "assistant", "content": response.content})

        # Anything other than tool_use means the model is done (end_turn),
        # cut off (max_tokens) or declined (refusal) - stop looping.
        if response.stop_reason != "tool_use":
            return messages

        # The model may ask for several tools at once. Run them all and send
        # every result back in ONE user message, each matched by tool_use_id.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                output, is_error = run_tool(block.name, block.input)
                print(f"  [tool_result] {output}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                })
        messages.append({"role": "user", "content": tool_results})

    print(f"\n!! Stopped after max_turns={max_turns} (runaway-loop guard)")
    return messages


if __name__ == "__main__":
    save_output(__file__)  # everything printed also goes to step2/agent.txt
    run_agent("What's Alice's balance?")
