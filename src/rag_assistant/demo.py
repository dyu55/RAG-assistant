"""Fictional, distributable documents used by the demo and evaluation suite."""

DOCUMENTS = {
    "Atlas architecture.md": """# Atlas / System architecture

Atlas is a fictional delivery platform used to demonstrate evidence retrieval.
Atlas uses Postgres as the source of truth for delivery orders. Redis stores a short-lived copy of frequently requested delivery status.
The Gateway authenticates requests before forwarding them to Atlas. The Worker processes jobs from the dispatch queue and records the result in Postgres.
Atlas publishes delivery events to the Event Bus. Analytics consumes events to build daily delivery reports.

## Operating boundaries
Redis is an optional acceleration layer. Atlas must remain available when Redis is offline. Redis must never be the only place where a delivery order is stored.
Postgres backups are retained for thirty days. The Recovery Service verifies a restore every week.
""",
    "Cache operations.md": """# Atlas / Cache operations

## Redis outage response
When Redis is unavailable, Atlas bypasses the cache and reads delivery status directly from Postgres. A circuit breaker prevents repeated connection attempts to Redis during an outage.
The on-call engineer checks the Redis health dashboard and monitors Postgres connection usage. After Redis recovers, Atlas gradually warms the cache to avoid a sudden load spike.

## Freshness
Redis delivery-status entries expire after sixty seconds. Atlas invalidates a cached status when a delivery update is committed to Postgres.
The Gateway rate limits expensive requests during cache recovery. The Worker continues processing delivery jobs while the cache is unavailable.
""",
    "Reliability playbook.md": """# Atlas / Reliability playbook

The Observability Service tracks request latency, error rates, and cache availability for Atlas. A Redis outage triggers an alert to the on-call engineer.
The Incident Commander coordinates recovery and records the incident timeline. Customer Support publishes service updates after the Incident Commander confirms the customer impact.
The Recovery Service restores Postgres from a verified backup during a database recovery exercise. Restore exercises run weekly; backups are retained for thirty days.
Each incident review records the cause, corrective actions, and an owner for each follow-up. The Engineering Lead reviews open actions every Friday.
""",
    "平台说明.txt": """Atlas 是用于演示知识检索的虚构配送平台。
Redis 故障时，Atlas 会绕过缓存，直接从 Postgres 读取配送状态。
Redis 恢复后，系统会逐步预热缓存，避免数据库负载突然增加。
Postgres 备份保留三十天，每周进行一次恢复演练。
""",
}

QUESTIONS = [
    ("How does Atlas handle a Redis outage?", "Cache operations.md"),
    ("How long are Postgres backups retained?", "Atlas architecture.md"),
    ("Who coordinates recovery during an incident?", "Reliability playbook.md"),
    ("How often do Redis delivery status entries expire?", "Cache operations.md"),
    ("Redis 故障时如何读取配送状态？", "平台说明.txt"),
]
