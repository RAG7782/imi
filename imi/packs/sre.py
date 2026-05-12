"""SRE Memory Pack — 500 pre-generated SRE incidents.

Provides a ready-to-use IMISpace pre-loaded with realistic SRE incidents
covering 10 pattern types across 180 simulated days.

Usage:
    from imi.packs.sre import load_sre_pack, SRE_INCIDENTS

    # Load into an IMISpace (encodes all 500 incidents)
    space = load_sre_pack("sre_memory.db")

    # Or just get the raw incidents
    for incident in SRE_INCIDENTS[:5]:
        print(incident["text"])
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from imi.space import IMISpace

# ---------------------------------------------------------------------------
# Incident generator (extended from AMBench with more variety)
# ---------------------------------------------------------------------------

TEMPLATES = {
    "connection_pool": {
        "templates": [
            "Connection pool exhaustion on {service}: {pool_size} connections maxed out during {trigger}",
            "Database connection pool depleted on {service} after {trigger}, all queries timing out at {latency}ms",
            "{service} unable to acquire DB connection, pool at max {pool_size}, {pct}% queries failing",
            "Connection pool leak on {service}: {pool_size} connections allocated but only {active} active",
            "PostgreSQL connection limit reached on {service}: max_connections={pool_size}, {count} waiting",
        ],
        "runbook": "Check connection pool metrics, identify leaking connections, restart if needed. Consider pgbouncer.",
        "severity_dist": [0.1, 0.3, 0.4, 0.2],  # low, medium, high, critical
    },
    "memory_leak": {
        "templates": [
            "Memory leak in {service}: RSS grew from {start_mb}MB to {end_mb}MB over {hours}h",
            "{service} OOM killed after memory usage reached {end_mb}MB, leak in {component}",
            "Gradual memory increase in {service} {component}: {start_mb}MB to {end_mb}MB in {hours}h",
            "Java heap exhaustion on {service}: -Xmx{start_mb}m insufficient, GC overhead at {pct}%",
            "Container memory limit breached on {service}: {end_mb}MB used of {start_mb}MB limit",
        ],
        "runbook": "Capture heap dump, analyze with MAT/jmap, identify object retention. Apply fix or increase limits.",
        "severity_dist": [0.05, 0.2, 0.45, 0.3],
    },
    "timeout_cascade": {
        "templates": [
            "Timeout cascade: {service_a} -> {service_b} -> {service_c}, p99 latency {latency}ms",
            "Request timeout chain: {service_a} (30s) -> {service_b} (30s) -> {service_c} ({latency}ms)",
            "Cascading timeouts from {service_a}: {service_b} and {service_c} both returning 504s",
            "Circuit breaker open on {service_a}: downstream {service_b} p99={latency}ms (threshold: 500ms)",
            "Retry storm between {service_a} and {service_b}: {rps} req/s amplified to {rps_amplified} req/s",
        ],
        "runbook": "Check downstream health, enable circuit breakers, add request deadlines. Investigate root cause.",
        "severity_dist": [0.05, 0.15, 0.4, 0.4],
    },
    "cert_expiry": {
        "templates": [
            "TLS cert expired on {service}: {affected_count} clients getting handshake failures",
            "Certificate rotation failed on {service}, manual renewal needed for *.{domain}",
            "Intermediate CA cert expired affecting {service}: {affected_count} downstream services impacted",
            "Let's Encrypt renewal failed for {domain}: DNS-01 challenge timeout after {duration}m",
            "mTLS client cert expired on {service}: {affected_count} service-to-service calls failing",
        ],
        "runbook": "Renew cert manually (certbot/acme), verify chain, reload service. Set up automated renewal.",
        "severity_dist": [0.1, 0.3, 0.4, 0.2],
    },
    "deploy_rollback": {
        "templates": [
            "Deployment of {service} v{version} rolled back: {error_type} errors at {pct}%",
            "{service} v{version} canary failed: {error_type} rate {pct}% exceeds 5% threshold",
            "Blue-green deploy of {service} v{version} aborted: health check failing on new pods",
            "Rolling update of {service} stuck: {pct}% pods in CrashLoopBackOff after v{version}",
            "Feature flag rollback on {service}: {error_type} errors spiked {pct}% after enabling {component}",
        ],
        "runbook": "Rollback via kubectl/helm, check logs, identify breaking change. Run integration tests.",
        "severity_dist": [0.2, 0.3, 0.35, 0.15],
    },
    "dns_failure": {
        "templates": [
            "DNS resolution failure for {service}.internal: {affected_count} services impacted {duration}m",
            "CoreDNS pod restart caused {duration}m of NXDOMAIN for {service} lookups",
            "DNS cache poisoning: {service} resolving to wrong IP, {affected_count} requests misdirected",
            "Split-brain DNS: {service} resolving differently in us-east vs eu-west for {duration}m",
            "Route53 health check flapping on {service}: {affected_count} failovers in {duration}m",
        ],
        "runbook": "Check DNS pods, flush caches, verify records. For cache poisoning: flush and update DNSSEC.",
        "severity_dist": [0.05, 0.15, 0.35, 0.45],
    },
    "disk_full": {
        "templates": [
            "Disk full on {service}: {path} at 100%, writes failing",
            "{service} disk space exhausted on {path}: {size}GB logs accumulated, rotation failed",
            "EBS volume full on {service}: {size}GB, auto-expansion failed due to IAM permissions",
            "Kafka log.dirs full on {service}: {size}GB, partitions going offline",
            "Elasticsearch index exceeds disk watermark on {service}: {size}GB, shard allocation blocked",
        ],
        "runbook": "Clean old logs/data, expand volume, fix log rotation. For Kafka: increase retention policy.",
        "severity_dist": [0.15, 0.35, 0.35, 0.15],
    },
    "rate_limit": {
        "templates": [
            "Rate limiter on {service}: {rps} req/s exceeded, {pct}% rejected",
            "{service} returning 429s: {rps} req/s from {trigger}, rate limit at {rate_limit} req/s",
            "API gateway throttling {service}: {pct}% of requests rate-limited during {trigger}",
            "Redis-backed rate limiter on {service} failing: {rps} req/s getting through unthrottled",
            "Distributed rate limiter race condition on {service}: allowing {rps} req/s vs {rate_limit} limit",
        ],
        "runbook": "Check traffic source, adjust limits if legitimate. For broken limiter: fix Redis/fallback.",
        "severity_dist": [0.2, 0.4, 0.3, 0.1],
    },
    "data_inconsistency": {
        "templates": [
            "Data inconsistency: {service_a} and {service_b} diverged on {count} records",
            "Replication lag {service_a} -> {service_b}: {count} records {duration}m behind",
            "Dual-write conflict: {service_a} and {service_b} have different values for {count} entities",
            "CDC pipeline lag: {service_a} changelog {count} events behind {service_b}",
            "Eventual consistency violation: {service_a} serving stale reads, {count} records affected",
        ],
        "runbook": "Run reconciliation job, identify source of divergence, fix replication pipeline.",
        "severity_dist": [0.1, 0.3, 0.4, 0.2],
    },
    "auth_failure": {
        "templates": [
            "Auth failure spike on {service}: {pct}% of requests getting 401",
            "JWT validation failing on {service}: key rotation left old keys, {pct}% errors",
            "OAuth token refresh race on {service}: {pct}% of users locked out {duration}m",
            "OIDC provider timeout: {service} unable to validate tokens, {affected_count} users affected",
            "Session store ({component}) on {service} returning stale sessions, {pct}% auth failures",
        ],
        "runbook": "Check auth provider, key rotation status, session store health. May need manual key sync.",
        "severity_dist": [0.1, 0.25, 0.4, 0.25],
    },
}

SERVICES = [
    "api-gateway",
    "user-service",
    "payment-service",
    "order-service",
    "inventory-service",
    "notification-service",
    "analytics-service",
    "auth-service",
    "search-service",
    "cdn-edge",
    "billing-service",
    "recommendation-engine",
    "data-pipeline",
    "scheduler-service",
    "config-service",
]

TRIGGERS = [
    "Black Friday traffic",
    "batch job spike",
    "DDoS attempt",
    "marketing campaign launch",
    "data migration",
    "partner API burst",
    "end-of-month reconciliation",
    "automated security scan",
    "load test runaway",
    "cron job overlap",
]

COMPONENTS = [
    "request handler",
    "cache layer",
    "session store",
    "message queue consumer",
    "background worker",
    "gRPC server",
    "connection pool manager",
    "rate limiter",
    "health check endpoint",
]

DOMAINS = ["api.example.com", "auth.internal", "payments.prod", "data.analytics.io"]

SEVERITIES = ["low", "medium", "high", "critical"]


def _generate_sre_incidents(n: int = 500, days: int = 180, seed: int = 42) -> list[dict]:
    """Generate n SRE incidents over `days` simulated days."""
    rng = random.Random(seed)
    incidents = []
    pattern_types = list(TEMPLATES.keys())

    # Power-law pattern distribution
    weights = {p: rng.paretovariate(0.8) for p in pattern_types}
    total = sum(weights.values())
    weights = {p: w / total for p, w in weights.items()}

    for i in range(n):
        # Pick pattern
        roll = rng.random()
        cumul = 0
        pattern = pattern_types[0]
        for p, w in weights.items():
            cumul += w
            if roll <= cumul:
                pattern = p
                break

        tpl = rng.choice(TEMPLATES[pattern]["templates"])
        sev_dist = TEMPLATES[pattern]["severity_dist"]

        params = {
            "service": rng.choice(SERVICES),
            "service_a": rng.choice(SERVICES),
            "service_b": rng.choice(SERVICES),
            "service_c": rng.choice(SERVICES),
            "pool_size": rng.choice([50, 100, 200, 500, 1000]),
            "active": rng.randint(10, 50),
            "trigger": rng.choice(TRIGGERS),
            "start_mb": rng.randint(256, 2048),
            "end_mb": rng.randint(2048, 16384),
            "hours": rng.randint(1, 72),
            "latency": rng.choice([500, 1000, 2000, 5000, 10000, 30000]),
            "affected_count": rng.randint(3, 200),
            "version": f"{rng.randint(1, 8)}.{rng.randint(0, 30)}.{rng.randint(0, 99)}",
            "error_type": rng.choice(["5xx", "timeout", "connection_refused", "OOM", "segfault"]),
            "pct": rng.randint(5, 95),
            "duration": rng.randint(2, 180),
            "rps": rng.choice([100, 500, 1000, 5000, 10000, 50000]),
            "rps_amplified": rng.choice([5000, 10000, 50000, 100000]),
            "rate_limit": rng.choice([100, 500, 1000, 5000]),
            "path": rng.choice(["/var/log", "/data", "/tmp", "/opt/app/logs", "/mnt/storage"]),
            "size": rng.randint(20, 2000),
            "count": rng.randint(10, 100000),
            "component": rng.choice(COMPONENTS),
            "domain": rng.choice(DOMAINS),
        }

        try:
            text = tpl.format(**params)
        except KeyError:
            text = tpl  # fallback

        day = int(i / n * days)
        severity = rng.choices(SEVERITIES, weights=sev_dist, k=1)[0]

        incidents.append(
            {
                "id": f"sre_{i:04d}",
                "text": text,
                "pattern_type": pattern,
                "day": day,
                "severity": severity,
                "runbook": TEMPLATES[pattern]["runbook"],
                "tags": [pattern, severity, params.get("service", "")],
            }
        )

    return incidents


# Pre-generate the pack
SRE_INCIDENTS = _generate_sre_incidents(500, 180, seed=42)


def load_sre_pack(db_path: str = "sre_memory.db", n: int | None = None) -> "IMISpace":
    """Load the SRE memory pack into an IMISpace.

    Args:
        db_path: Path for the SQLite database
        n: Number of incidents to load (default: all 500)

    Returns:
        IMISpace pre-loaded with SRE incidents
    """
    from imi.space import IMISpace

    space = IMISpace.from_sqlite(db_path)

    # Skip if already loaded
    if len(space.episodic) > 0:
        return space

    incidents = SRE_INCIDENTS[:n] if n else SRE_INCIDENTS

    for inc in incidents:
        space.encode(
            inc["text"],
            tags=inc["tags"],
            source="sre-pack",
            context_hint=f"Day {inc['day']}, severity: {inc['severity']}. Runbook: {inc['runbook']}",
        )

    return space
