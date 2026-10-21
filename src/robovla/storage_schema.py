"""Storage seam schema (design §11: the single DDL for the console state DB).

Lives with :mod:`robovla.storage` (the only SQL in the codebase is storage.py + this
schema text). Tables: users, sessions, tokens, agent_tokens, datasets, chunks,
tickets, jobs, artifacts, audit, idempotency, backups, kv (design §8/§9).
"""

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    role TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 0,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL DEFAULT '',
    email TEXT NOT NULL DEFAULT '',
    max_running INTEGER,
    max_queued INTEGER,
    max_dataset_bytes INTEGER,
    max_run_min INTEGER,
    run_now INTEGER,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
    token_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(user_id),
    token_hash TEXT NOT NULL UNIQUE,
    -- sha256 of the raw secret: the O(1) find path (the argon2 hash stays
    -- authoritative; §9.8 shown-once tokens are verified hash-then-confirm)
    token_lookup TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    revoked_at TEXT
);
CREATE TABLE IF NOT EXISTS agent_tokens (
    node_id TEXT PRIMARY KEY,
    token_hash TEXT NOT NULL,
    token_lookup TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    rotated_at TEXT
);
CREATE TABLE IF NOT EXISTS datasets (
    dataset_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    status TEXT NOT NULL,
    total_bytes INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    qa_json TEXT,
    norm_stats_json TEXT,
    split_json TEXT,
    rejected_reason TEXT
);
CREATE TABLE IF NOT EXISTS chunks (
    dataset_id TEXT NOT NULL REFERENCES datasets(dataset_id),
    chunk_index INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    nbytes INTEGER NOT NULL,
    PRIMARY KEY (dataset_id, chunk_index)
);
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    owner TEXT NOT NULL,
    hypothesis TEXT NOT NULL,
    config_summary TEXT NOT NULL,
    prediction TEXT NOT NULL,
    linked_job TEXT,
    outcome TEXT,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    spec_json TEXT NOT NULL,
    state TEXT NOT NULL,
    node_id TEXT,
    submitted_at REAL NOT NULL,
    run_now INTEGER NOT NULL DEFAULT 0,
    ticket_id TEXT,
    idempotency_key TEXT UNIQUE,
    created_at TEXT NOT NULL,
    terminal_at TEXT,
    note TEXT,
    slot_handle TEXT,
    last_used_at TEXT,
    predict_probe_json TEXT,
    run_id TEXT
);
CREATE INDEX IF NOT EXISTS jobs_student_state ON jobs (state, node_id);
CREATE TABLE IF NOT EXISTS artifacts (
    job_id TEXT NOT NULL REFERENCES jobs(job_id),
    kind TEXT NOT NULL,
    path TEXT NOT NULL,
    nbytes INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (job_id, kind)
);
CREATE TABLE IF NOT EXISTS audit (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    target TEXT NOT NULL,
    before_json TEXT,
    after_json TEXT
);
CREATE INDEX IF NOT EXISTS audit_at ON audit (at);
CREATE TABLE IF NOT EXISTS idempotency (
    key TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS backups (
    at TEXT PRIMARY KEY,
    dest TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    sha256 TEXT NOT NULL,
    status TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS kv (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""
