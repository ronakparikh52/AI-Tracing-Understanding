# Mini-Bank: building an AI agent from scratch, one step at a time

A hands-on sandbox for learning how AI agents actually work under the hood —
no frameworks (no LangGraph, CrewAI, PydanticAI), just Python, the official
[Anthropic SDK](https://github.com/anthropics/anthropic-sdk-python) and a local
Postgres "bank".

The agent is a customer-support bot for a tiny fake bank. It can look up
accounts and transactions and issue refunds — which makes it a good testbed for
the hard parts of real agents: tool calling, guardrails, prompt injection,
prompt caching, non-determinism ("drift"), evals and tracing.

Each `stepN/` folder adds **one** idea on top of the previous step. Old steps
are never edited, so you can diff any two steps to see exactly what changed.
Each step's real outputs (`.txt` / `.jsonl`) are saved next to its code, so you
can read the results without running anything (or spending API credits).

---

## The sandbox

**The bank** (`seed.sql`): 3 customers and 4 transactions, each one planted to
test a different edge case.

| Account | Transaction | Status | Tests |
|---|---|---|---|
| 1 Alice ($500) | Acme Cloud Hosting $49 | completed | the clean, should-refund case |
| 1 Alice | Coffee Roasters $18.50 | disputed, note: "maybe my roommate?" | ambiguity — the agent should ask |
| 2 Bob ($120) | FlightBooker Pro $350 | completed, "non-refundable" | over the $100 auto-refund limit |
| 3 Charlie ($25) | StreamFlix $15 | already refunded | double-refund attempt |

Policies (also in the DB): auto-refunds up to **$100**, no double refunds, ask
before refunding an ambiguous dispute.

**Shared files**

| File | What it is |
|---|---|
| `docker-compose.yml` | Postgres 16 on **localhost:5433** (5433 so it won't clash with a local Postgres) |
| `seed.sql` | Schema + seed data, loaded automatically on first boot |
| `db.py` | Tiny DB helper: `query()`, `execute()`, `reset_db()`, and (step 7) template-DB branching; (step 7b) the active DB is a `ContextVar`, so each thread talks to its own branch |
| `common.py` | `save_output()` (mirror prints to a `.txt`), `JsonlLog` (raw JSON log), and (step 7) `Tracer` for spans; (step 7b) thread-safe: locks around file writes, a per-thread span stack |

---

## Setup

Requirements: Docker (or Colima), Python 3.10+, and an Anthropic API key
(Console credits — a full run of every step costs a few dollars at most:
step 7b ≈ $0.40, step 8 ≈ $0.70).

```bash
# 1. Start the bank
docker compose up -d --wait
docker compose exec postgres psql -U bank -d minibank -c "SELECT * FROM accounts;"

# 2. Python env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Auth: either export ANTHROPIC_API_KEY=... or log in with the `ant` CLI

# 4. Run any step from the project root
python step1/agent.py
```

Models: steps use `claude-haiku-5-5` (cheap); steps 7/7b also compare
`claude-sonnet-5-5`.

---

## The steps

### Step 1 — one API call
`step1/agent.py` · output: `agent.txt`

A single `messages.create()` with no tools. Asked for Alice's balance, the
model correctly says it can't know. **Lesson:** the Messages API is
**stateless** — the model only knows what's in the request. Shows the response
shape (content blocks, `stop_reason`, token `usage`).

### Step 2 — one tool + the agent loop
`step2/agent.py` · output: `agent.txt`

Adds a `get_account` tool and **the agent loop**: send the conversation + tool
definitions → if `stop_reason == "tool_use"`, *your code* runs the tool →
append the result → call again → until the model stops asking. That loop *is*
an agent. **Lessons:** tool schemas (`input_schema`, `strict: true`), the
`tool_use` → `tool_result` round trip, and that tool definitions cost tokens
(~480 here) on every call.

### Step 3 — several tools, including one that writes money
`step3/agent.py` · outputs: `<scenario>.txt` (readable) + `<scenario>.jsonl` (raw)

Adds `get_transactions`, `get_refund_policies` and a deliberately **naive**
`issue_refund` that does whatever it's asked. The *only* safety is the model
reading the policies. Four scenarios (clean / flight / double / ambiguous);
the DB is reset before each. **Result:** all four decisions were right — but the
agent reported a *stale* balance after refunding. **Lesson:** "it worked" on
judgment alone is not a guarantee.

### Step 4 — guardrails in code, not in the prompt
`step4/agent.py`, `step4/attack_prompt_only.py`

Defense in depth:
- **Session** — who's logged in comes from auth, never from the chat. Tools take
  no account/name arguments, so the model *can't express* "look at someone
  else's account".
- **Rules enforced inside the tool** (`PolicyError`): ownership, no double
  refunds, the $100 limit — checked in code with a `SELECT … FOR UPDATE` row lock.
- **Human approval gate** before money moves (`--auto-approve` to skip).
- Tools return **post-write state** (`new_balance_cents`) — fixes the stale balance.

Three attacks are added: a fake "manager approved it" claim, a **prompt
injection** planted in a transaction note, and **impersonation** ("Hi, I'm Alice"
while logged in as Charlie). `attack_prompt_only.py` fires the same attacks at
step 3's agent: **impersonation succeeded against step 3** (it refunded Alice's
charge to Charlie's request) and was blocked in step 4.

### Step 5 — multi-turn conversations
`step5/agent.py` · outputs per scenario + `summary.txt`

Two loops now: an **outer conversation loop** (one iteration per user message)
around the **inner agent loop** (one iteration per API call). The key line is
appending each new user message to the *same* `messages` list — the whole
history is re-sent every call, so input tokens grow every turn. Scenarios:
clarify-then-refund, clarify-then-don't, topic switch, and a 3-turn pushback
attack. `--chat --as 1` chats live. `summary.txt` gives one block per scenario —
read it first.

### Step 6 — prompt caching, measured
`step6/agent.py` · outputs per experiment + `summary.txt`

Because every call re-sends the whole history, most of each request is
identical to the last. `cache_control` breakpoints let the API reuse that
already-processed prefix (the KV cache) at 0.1× the price. Five experiments on
the same conversation:

| Experiment | What it does | Result (Haiku) |
|---|---|---|
| `no_cache` | no breakpoints | 0% cached (baseline) |
| `system_only` | breakpoint after tools + system | 48% cached, −33% cost |
| `full` | + top-level automatic breakpoint that moves with the conversation | 80% cached, −54% |
| `cross_user` | a *different customer* reusing the same prefix | cache hit on call 1 |
| `busted` | a timestamp at the start of the system prompt | 0% cached, **+20%** cost |

**Lessons:** caches match on exact prefix bytes in order tools → system →
messages; anything volatile placed early breaks everything after it.

### Step 7 — drift: does it pass *every* time?
`step7/agent.py` · outputs: `summary.txt`, `failures.txt`, `traces.jsonl`, `api_calls.jsonl`, `run.txt`

The agent is unchanged; this step is an **eval harness** around it. All 11
scenarios from steps 4–5 run repeatedly (Haiku ×20, Sonnet ×5) and are graded
automatically.
- **Auto-grading** on the DB end state: each scenario declares the *exact*
  changes a correct run makes (e.g. `clean` → Alice $549, txn 1 refunded).
- **pass^k** — the estimated chance that *k customers in a row* all get the
  right outcome. 95% per run sounds good; pass^5 is only ~75%. For a bank, that
  is the number that matters.
- **Span tracing** — every run is a trace tree (`run → user_turn → api_call /
  tool`) with timings, tokens and cost, using OpenTelemetry's field names.
  Failed runs get their tree drawn in `failures.txt`.
- **Database branching** — each run gets its own clone of a clean template DB
  (`CREATE DATABASE … TEMPLATE`). Passed runs' branches are dropped; **failed
  runs' branches are kept** so you can `psql` in and see exactly what happened.

```bash
python step7/agent.py --models haiku --n 2 flight   # cheap smoke test
```

Step 7 runs one conversation at a time, and the full run was **stopped
partway** — it was going to take ~25 minutes. Its saved outputs (`run.txt`,
`traces.jsonl`, `api_calls.jsonl`) are from that partial run. The traces showed
why it was slow: ~97% of each run is waiting on the API (~1.6 s per call vs ~32 ms
per tool). The full results come from step 7b.

### Step 7b — the same eval, run in parallel
`step7b/agent.py` · outputs: `summary.txt`, `failures.txt`, `traces.jsonl`, `api_calls.jsonl`, `run.txt`

Identical scenarios and grading; runs execute **10 at a time** with a
`ThreadPoolExecutor`. Threads fit here because the work is I/O-bound (waiting on
HTTP), so Python's GIL isn't a bottleneck. What had to change to make it safe:
- **No shared mutable globals** — the model and its prices are passed down
  explicitly instead of living in module-level variables.
- **Per-thread state via `ContextVar`** — each thread has its own active DB
  branch (`db.py`) and its own span stack (`Tracer`), so parallel runs never
  become each other's parent spans or write to each other's database.
- **Locks** around shared files (`JsonlLog`, `traces.jsonl`) and around
  `CREATE DATABASE … TEMPLATE` (Postgres refuses concurrent clones of one template).
- **Retries** — `anthropic.Anthropic(max_retries=8)` backs off on 429 rate limits.

**Results:** 275/275 runs passed (Haiku ×20 and Sonnet ×5 on all 11 scenarios),
pass^5 = 100% everywhere. ~9.5× faster than sequential (153 s wall-clock vs
1,458 s of summed run time). Haiku cost $0.09 for 220 runs; Sonnet $0.31 for 55 —
**~12× more per run at equal accuracy**. None of the attacks even reached the
code guardrails: the model refused first. 0 failures in 20 runs still only
bounds the failure rate at roughly 3/20 (the "rule of three"), so more runs
would be needed to claim real reliability.

```bash
python step7b/agent.py                              # full run (~$0.40, ~3 min)
python step7b/agent.py --models haiku --workers 5 flight
```

### Step 8 — context rot
`step8/agent.py` · outputs: `summary.txt`, `failures.txt`, `traces.jsonl`, `api_calls.jsonl`, `run.txt`

Does the agent forget its rules when a long history sits in front of the
request? Each run starts with a fake history of earlier support chats with the
same customer — realistic filler plus **distractors** (old assistant messages
that contradict today's policy, e.g. "refunds up to $500 are automatic"). Sizes
are measured with the free `count_tokens` endpoint. Three variables:

| Variable | Values |
|---|---|
| History size | 0, ~20K, ~75K tokens (kept under Haiku's 100K price jump) |
| Where the rules live | `tool` — fetched via `get_refund_policies`, landing at the fresh end of the context; `system` — only in the system prompt, buried under the history |
| Strategy | `full` (send everything) or `trim` (sliding window: keep the last 4 real user turns, cutting only where `tool_use`/`tool_result` pairs stay intact) |

Two grades per run: **outcome** (DB end state right) and **judgment** (outcome
right *and* the model never tried a refund the code had to block — guardrails can
hide rot, so this checks whether the model itself still knows the rules). The
padding has its own cache breakpoint, so 98% of input tokens were cache reads.

**Results (Haiku, 420 runs, $0.69):** 419/420 passed. **No rule-following rot
at ~77K tokens** in either placement, and zero blocked refund attempts. The single
failure (`system_75k` / `topic_switch`) was over-caution: the model "corrected"
the distractor history and then asked for confirmation instead of refunding.
Long context still cost: latency per call grew ~1.9 s → 2.2 s → 2.8 s and cost
per run ~8×. Trimming brought cost back down 5× ($0.0034 → $0.0007 per run) with
no accuracy loss — the trade-off is that the trimmed prefix shifts every turn, so
the message part of the cache keeps getting rewritten.

```bash
python step8/agent.py                                      # all conditions, Haiku (~$0.70)
python step8/agent.py --n 2 --conditions tool_0k,system_75k flight
```

---

## Concepts map

| Concept | Where |
|---|---|
| Stateless API, response shape, tokens | step 1 |
| Tool schemas, the agent loop, `stop_reason` | step 2 |
| Write tools, scenarios, raw JSON logs | step 3 |
| Guardrails, sessions, approval gates, prompt injection | step 4 |
| Multi-turn state, context growth | step 5 |
| Prompt caching / KV-cache reuse, cost | step 6 |
| Non-determinism, evals, pass^k, tracing, DB branching | step 7 |
| Parallelism: threads, `ContextVar`, locks, retries | step 7b |
| Context rot, distractors, policy placement, trimming vs caching | step 8 |

## Roadmap
- **Step 9 — real tracing:** swap the hand-rolled tracer for OpenTelemetry and
  send spans to a backend.
