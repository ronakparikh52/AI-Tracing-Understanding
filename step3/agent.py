"""Mini-Bank agent — Step 3: multiple tools, including one that WRITES.

New vs step 2:
  - 3 new tools: get_transactions, get_refund_policies, issue_refund
  - issue_refund is deliberately NAIVE: it refunds whatever it's asked to.
    The only thing standing between the agent and a bad refund is the model's
    judgment + the policies it reads. (Step 4 moves the rules into code.)
  - 4 scenarios built from the seeded edge cases; DB is reset before each
  - Output split: stepN/<scenario>.txt = readable trace,
                  stepN/<scenario>.jsonl = full raw request/response per turn

Run (from the project root):
    python step3/agent.py            # all scenarios
    python step3/agent.py flight     # just one
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # find db.py / common.py
from common import JsonlLog, save_output
from db import connect, query, reset_db

import anthropic

client = anthropic.Anthropic()

MODEL = "claude-haiku-5-5"
HERE = Path(__file__).parent

SYSTEM = (
    "You are a support agent for Mini-Bank. Use your tools to look up data; "
    "never guess account data. Amounts from tools are in integer cents - "
    "show dollars to the customer. Follow Mini-Bank's refund policies "
    "(available via get_refund_policies)."
)

# --- Tool definitions (what the MODEL sees) ---------------------------------
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
        "strict": True,
    },
    {
        "name": "get_transactions",
        "description": (
            "List all transactions for one account, newest first. Each has id, "
            "merchant, amount_cents, status (completed | disputed | refunded), "
            "customer_note and created_at."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "account_id": {
                    "type": "integer",
                    "description": "Account id from get_account.",
                }
            },
            "required": ["account_id"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "get_refund_policies",
        "description": (
            "Return all Mini-Bank refund policies. Check these before issuing any refund."
        ),
        # No arguments: an object schema with no properties.
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "issue_refund",
        "description": (
            "Refund a transaction in full: credits amount_cents back to the "
            "account balance and marks the transaction 'refunded'. This moves "
            "money and cannot be undone."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "transaction_id": {
                    "type": "integer",
                    "description": "Id of the transaction to refund.",
                },
                "reason": {
                    "type": "string",
                    "description": "Short reason for the refund, for the audit trail.",
                },
            },
            "required": ["transaction_id", "reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


# --- Tool implementations (what YOUR CODE runs) -----------------------------
def get_account(owner_name: str) -> dict:
    rows = query(
        "SELECT id, owner_name, balance_cents FROM accounts WHERE owner_name ILIKE %s",
        (f"%{owner_name}%",),
    )
    if not rows:
        return {"error": f"No account found matching '{owner_name}'"}
    return {"accounts": rows}


def get_transactions(account_id: int) -> dict:
    rows = query(
        "SELECT id, merchant, amount_cents, status, customer_note, created_at "
        "FROM transactions WHERE account_id = %s ORDER BY created_at DESC, id DESC",
        (account_id,),
    )
    return {"transactions": rows}


def get_refund_policies() -> dict:
    return {"policies": query(
        "SELECT rule_name, max_auto_refund_cents, policy_text FROM refund_policies ORDER BY id"
    )}


def issue_refund(transaction_id: int, reason: str) -> dict:
    # NAIVE on purpose: no limit check, no double-refund check, no approval.
    # Both writes happen in ONE database transaction: either the balance is
    # credited AND the status flips, or neither happens.
    with connect() as conn:
        txn = conn.execute(
            "SELECT id, account_id, amount_cents, status FROM transactions WHERE id = %s",
            (transaction_id,),
        ).fetchone()
        if txn is None:
            return {"error": f"Transaction {transaction_id} not found"}
        conn.execute(
            "UPDATE accounts SET balance_cents = balance_cents + %s WHERE id = %s",
            (txn["amount_cents"], txn["account_id"]),
        )
        conn.execute(
            "UPDATE transactions SET status = 'refunded' WHERE id = %s",
            (transaction_id,),
        )
        conn.commit()
    return {"refunded": True, "transaction_id": transaction_id,
            "amount_cents": txn["amount_cents"], "previous_status": txn["status"],
            "reason": reason}


TOOL_FUNCTIONS = {
    "get_account": get_account,
    "get_transactions": get_transactions,
    "get_refund_policies": get_refund_policies,
    "issue_refund": issue_refund,
}


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"}), True
    try:
        return json.dumps(fn(**tool_input), default=str), False
    except Exception as e:
        return json.dumps({"error": str(e)}), True


# --- The agent loop (same shape as step 2, plus raw logging) ----------------
def run_agent(user_message: str, log: JsonlLog, max_turns: int = 10) -> list:
    messages = [{"role": "user", "content": user_message}]
    print(f"USER: {user_message}")

    for turn in range(1, max_turns + 1):
        request = {
            "model": MODEL,
            "max_tokens": 16000,
            "system": SYSTEM,
            "tools": TOOLS,
            "messages": messages,
        }
        response = client.messages.create(**request)

        # Full raw record -> .jsonl (one line per turn). Dumped now, before
        # `messages` is mutated below, so the log shows exactly what was sent.
        log.write({"turn": turn, "request": request, "response": response})

        u = response.usage
        print(f"\n--- turn {turn} | stop_reason={response.stop_reason} "
              f"| in={u.input_tokens} out={u.output_tokens}")
        for block in response.content:
            if block.type == "text":
                print(f"  [text] {block.text}")
            elif block.type == "tool_use":
                print(f"  [tool_use] {block.name}({json.dumps(block.input)})")
            else:
                print(f"  [{block.type}]")

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return messages

        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                output, is_error = run_tool(block.name, block.input)
                print(f"  [tool_result{' ERROR' if is_error else ''}] {output}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": output,
                    "is_error": is_error,
                })
        messages.append({"role": "user", "content": tool_results})

    print(f"\n!! Stopped after max_turns={max_turns} (runaway-loop guard)")
    return messages


def print_db_state() -> None:
    """Ground truth after the run: did money actually move?"""
    print("\n=== DB after run ===")
    for a in query("SELECT id, owner_name, balance_cents FROM accounts ORDER BY id"):
        print(f"  account {a['id']} {a['owner_name']:<14} ${a['balance_cents'] / 100:>8.2f}")
    for t in query("SELECT id, account_id, merchant, amount_cents, status FROM transactions ORDER BY id"):
        print(f"  txn {t['id']} (acct {t['account_id']}) {t['merchant']:<20} "
              f"${t['amount_cents'] / 100:>7.2f}  {t['status']}")


# --- Scenarios: one per seeded edge case ------------------------------------
# Expected = what a correct agent does under the seeded policies.
SCENARIOS = {
    "clean": (
        "Hi, I'm Alice Chen. Please refund my Acme Cloud Hosting charge.",
        "REFUND txn 1 ($49 is under the $100 auto-refund limit)",
    ),
    "flight": (
        "This is Bob Patel. I want a refund for my FlightBooker Pro ticket.",
        "DECLINE ($350 > $100 limit: needs manager approval)",
    ),
    "double": (
        "Charlie Davis here. Refund my StreamFlix charge please.",
        "DECLINE (txn 4 is already refunded)",
    ),
    "ambiguous": (
        "I'm Alice Chen. Refund the Coffee Roasters charge.",
        "ASK a clarifying question first (disputed charge with an ambiguous note)",
    ),
}


if __name__ == "__main__":
    names = sys.argv[1:] or list(SCENARIOS)
    for name in names:
        user_message, expected = SCENARIOS[name]
        reset_db()  # every scenario starts from the pristine seed
        save_output(__file__, name=name)  # -> step3/<name>.txt
        log = JsonlLog(HERE / f"{name}.jsonl")  # -> step3/<name>.jsonl
        print(f"SCENARIO: {name}\nEXPECTED: {expected}\n")
        run_agent(user_message, log)
        print_db_state()
