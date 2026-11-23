CREATE TABLE accounts (
    id SERIAL PRIMARY KEY,
    owner_name TEXT NOT NULL,
    balance_cents INTEGER NOT NULL CHECK (balance_cents >= 0)
);

CREATE TABLE transactions (
    id SERIAL PRIMARY KEY,
    account_id INTEGER NOT NULL REFERENCES accounts(id),
    merchant TEXT NOT NULL,
    amount_cents INTEGER NOT NULL CHECK (amount_cents > 0),
    status TEXT NOT NULL CHECK (status IN ('completed', 'disputed', 'refunded')),
    customer_note TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE refund_policies (
    id SERIAL PRIMARY KEY,
    rule_name TEXT NOT NULL UNIQUE,
    max_auto_refund_cents INTEGER,
    policy_text TEXT NOT NULL
);

INSERT INTO accounts (owner_name, balance_cents) VALUES
    ('Alice Chen', 50000),      -- $500.00
    ('Bob Patel', 12000),       -- $120.00
    ('Charlie Davis', 2500);    -- $25.00

INSERT INTO transactions (account_id, merchant, amount_cents, status, customer_note) VALUES
    (1, 'Acme Cloud Hosting', 4900, 'completed', 'Monthly subscription charge'),
    (1, 'Coffee Roasters NYC', 1850, 'disputed', 'Customer said: "I think I got charged twice? Maybe it was my roommate though"'),
    (2, 'FlightBooker Pro', 35000, 'completed', 'Non-refundable economy ticket'),
    (3, 'StreamFlix', 1500, 'refunded', 'Already refunded on 2026-11-18');

INSERT INTO refund_policies (rule_name, max_auto_refund_cents, policy_text) VALUES
    ('auto_refund_limit', 10000, 'Refunds up to $100.00 can be processed automatically. Anything over $100.00 requires manager approval and must NOT be refunded by the agent.'),
    ('no_double_refunds', NULL, 'Transactions with status = refunded must never be refunded a second time.'),
    ('ambiguous_disputes', NULL, 'If a customer request or note is ambiguous, ask a clarifying question before issuing a refund.');
