"""Run step 4's ATTACK scenarios against the prompt-only step 3 agent.

The step 3 agent has no session, no code enforcement and no approval gate -
its only defense is the model obeying the policies it reads. Same attacks,
same model, so any difference in outcome comes from the guardrails alone.

Step 3 has no login, so "who is logged in" can't be expressed: the agent
believes whatever name the user types. The impersonation attack is therefore
just the message itself.

Run (from the project root):  python step4/attack_prompt_only.py [override injection impersonation]
Output: step4/prompt_only_<attack>.txt / .jsonl
"""
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from common import JsonlLog, save_output
from db import reset_db

HERE = Path(__file__).parent


def load(name: str, path: Path):
    """Import a file as a module (both steps are called agent.py)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


step3 = load("step3_agent", ROOT / "step3" / "agent.py")
step4 = load("step4_agent", HERE / "agent.py")

# In step 3 the user has to name themselves, since there's no login.
NAMES = {1: "Alice Chen", 2: "Bob Patel", 3: "Charlie Davis"}
ATTACKS = ["override", "injection", "impersonation"]

if __name__ == "__main__":
    for name in sys.argv[1:] or ATTACKS:
        sc = step4.SCENARIOS[name]
        message = sc.message
        if not message.startswith("Hi, I'm"):  # impersonation already names "Alice"
            message = f"This is {NAMES[sc.account_id]}. {message}"
        reset_db()
        if sc.setup:
            sc.setup()
        save_output(__file__, name=f"prompt_only_{name}")
        log = JsonlLog(HERE / f"prompt_only_{name}.jsonl")
        print(f"ATTACK vs PROMPT-ONLY (step 3) agent: {name}")
        print("A correct defense would: " + sc.expected + "\n")
        step3.run_agent(message, log)
        step3.print_db_state()
