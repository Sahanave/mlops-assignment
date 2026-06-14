"""Prompt templates for the agent nodes.

The GENERATE_SQL_* prompts are consumed by `generate_sql_node` in graph.py via
`.format(schema=..., question=...)`. The VERIFY_* and REVISE_* prompts are
consumed by the nodes implemented in graph.py with the placeholders documented
on each template below.

Design notes (these matter for both quality AND serving cost):
- The *_SYSTEM strings are written as a stable, reusable prefix: identical bytes
  on every call, with no per-request data interpolated. With vLLM prefix caching
  enabled (see scripts/start_vllm.sh) this prefix is computed once and reused,
  so the dominant prompt cost becomes the schema + question, not the rules.
- Variable input (schema, question, prior attempt) goes in the *_USER message,
  after the stable system prefix.
- Prompts are kept short and ask for a single fenced ```sql block (generate /
  revise) or a single compact JSON object (verify) so outputs stay small -
  the workload is "short structured outputs", which is what the SLO assumes.
"""

# ---------------------------------------------------------------------------
# generate_sql
# ---------------------------------------------------------------------------

GENERATE_SQL_SYSTEM = """You are an expert data analyst who writes SQLite SQL.
You are given a database schema and an English question. Write ONE SQLite query
that answers the question.

Rules:
- Output ONLY the SQL, inside a single ```sql ... ``` code block. No prose.
- Use only tables and columns that appear in the schema. Quote identifiers with
  double quotes when they contain spaces or are reserved words.
- Return exactly the columns the question asks for - no extra columns, no
  SELECT * unless the question really asks for whole rows.
- Prefer explicit JOINs using the foreign keys shown in the schema.
- Do not modify data: SELECT statements only."""

# Available placeholders: {schema}, {question}
GENERATE_SQL_USER = """Schema:
{schema}

Question: {question}

Write the SQLite query."""


# ---------------------------------------------------------------------------
# verify  (vLLM call #2) -> the node parses {"ok": bool, "issue": str}
# ---------------------------------------------------------------------------

VERIFY_SYSTEM = """You are a careful reviewer of SQL query results. You are given
a question, the SQL that was run, and the result of running it. Decide whether
the result plausibly answers the question.

Mark it NOT ok (ok=false) when any of these clearly hold:
- The execution returned an ERROR.
- Zero rows were returned but the question implies at least one row should exist
  (e.g. "which", "list", "how many ... that ...", "the most ...").
- The returned columns plainly do not answer what was asked (e.g. the question
  asks for a name but only an id was returned, or an aggregate was asked for but
  raw rows came back).
- The result is obviously wrong in shape (e.g. many rows when a single value was
  asked for).

Otherwise mark it ok=true. Do not nitpick correct-looking results.

Respond with ONLY a JSON object on a single line:
{"ok": true or false, "issue": "<short reason, empty string if ok>"}"""

# Available placeholders: {question}, {sql}, {result}
VERIFY_USER = """Question: {question}

SQL:
{sql}

Result of running the SQL:
{result}

Return the JSON verdict."""


# ---------------------------------------------------------------------------
# revise  (vLLM call #3) -> same output contract as generate_sql
# ---------------------------------------------------------------------------

REVISE_SYSTEM = """You are an expert data analyst fixing a SQLite query that did
not satisfy a reviewer. You are given the schema, the question, the previous SQL,
the result it produced, and the reviewer's complaint. Produce a corrected query.

Rules:
- Output ONLY the corrected SQL, inside a single ```sql ... ``` code block. No prose.
- Directly address the reviewer's complaint; do not just resubmit the same query.
- Use only tables and columns that appear in the schema, quoting identifiers when
  needed.
- SELECT statements only - do not modify data."""

# Available placeholders: {schema}, {question}, {sql}, {result}, {issue}
REVISE_USER = """Schema:
{schema}

Question: {question}

Previous SQL:
{sql}

Result it produced:
{result}

Reviewer's complaint: {issue}

Write the corrected SQLite query."""
