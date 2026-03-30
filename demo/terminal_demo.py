#!/usr/bin/env python3
"""Terminal demo for recording GIF/asciinema.

Run:
    python demo/terminal_demo.py

Record with asciinema:
    asciinema rec demo.cast -c "python demo/terminal_demo.py"
    agg demo.cast demo.gif

Or with terminalizer:
    terminalizer record demo -c "python demo/terminal_demo.py"
    terminalizer render demo
"""

import sys
import time
import os

# Suppress logs for clean demo
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import logging
logging.disable(logging.CRITICAL)

def slow_print(text, delay=0.02):
    """Print with typing effect."""
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(delay)
    print()

def section(title):
    print()
    print(f"\033[1;35m{'='*60}\033[0m")
    slow_print(f"\033[1;35m  {title}\033[0m", 0.03)
    print(f"\033[1;35m{'='*60}\033[0m")
    print()
    time.sleep(0.5)

def cmd(text):
    slow_print(f"\033[1;32m$ \033[0;36m{text}\033[0m", 0.02)
    time.sleep(0.3)

def output(text):
    print(f"  \033[0;37m{text}\033[0m")


def main():
    section("IMI — Integrated Memory Intelligence")
    slow_print("\033[3m  RAG finds what's similar. IMI finds what matters.\033[0m", 0.04)
    time.sleep(1)

    # --- Encode ---
    section("1. Encode: Agent learns from incidents")
    cmd("space.encode('DNS failure at 03:00 caused auth cascade')")
    time.sleep(0.5)

    from imi.space import IMISpace
    space = IMISpace()

    node = space.encode("DNS failure at 03:00 caused auth cascade across 3 services",
                        tags=["dns", "auth", "incident"])
    output(f"ID: {node.id}")
    output(f"Affect: salience={node.affect.salience:.1f} valence={node.affect.valence:.1f}")
    output(f"Mass: {node.mass:.2f} (critical incident — resists forgetting)")
    if node.affordances:
        output(f"Affordance: {node.affordances[0].action}")
    time.sleep(1)

    # More memories
    space.encode("Auth recovered after DNS fix at 03:45, 200 users lost sessions", tags=["auth"])
    space.encode("Database connection pool exhausted during peak load", tags=["database"])
    space.encode("Post-mortem: DNS failure root cause was misconfigured TTL", tags=["dns", "postmortem"])
    space.encode("Circuit breaker deployed on auth service", tags=["auth", "improvement"])
    output(f"\n  Total: {len(space.episodic)} memories encoded")
    time.sleep(1)

    # --- Navigate ---
    section("2. Navigate: Adaptive relevance weighting")

    queries = [
        ("recent auth failures", "TEMPORAL", 0.15),
        ("find all DNS incidents", "EXPLORATORY", 0.00),
        ("how to prevent cascades", "ACTION", 0.05),
    ]

    for q, expected_intent, expected_rw in queries:
        cmd(f"space.navigate('{q}')")
        rw, intent = space.adaptive_rw.classify_with_info(q)
        result = space.navigate(q, top_k=2)
        output(f"Intent: {intent.name} -> rw={rw}")
        for m in result.memories[:2]:
            score = m['score']
            content = m['content'][:70]
            output(f"  [{score:.2f}] {content}")
        print()
        time.sleep(0.8)

    # --- Graph ---
    section("3. Graph: Multi-hop causal chains")
    from imi.graph import EdgeType

    nodes = space.episodic.nodes
    dns = nodes[0]
    recovery = nodes[1]
    postmortem = nodes[3]

    space.graph.add_edge(dns.id, recovery.id, EdgeType.CAUSAL, label="caused")
    space.graph.add_edge(dns.id, postmortem.id, EdgeType.CAUSAL, label="investigated_in")

    cmd(f"space.graph.add_edge(dns, recovery, CAUSAL)")
    cmd(f"space.graph.add_edge(dns, postmortem, CAUSAL)")
    output(f"Graph: {space.graph.stats()['total_edges']} edges")
    print()

    cmd("space.navigate('what caused the auth outage?', use_graph=True)")
    result = space.navigate("what caused the auth outage?", use_graph=True, top_k=3)
    for m in result.memories[:3]:
        output(f"  [{m['score']:.2f}] {m['content'][:70]}")
    time.sleep(1)

    # --- Compare ---
    section("4. IMI vs RAG (pure cosine)")

    cmd("# RAG: relevance_weight=0.0 (pure cosine)")
    rag = space.navigate("recent auth issues", relevance_weight=0.0, top_k=2)
    for m in rag.memories[:2]:
        output(f"  [{m['score']:.2f}] {m['content'][:70]}")
    print()

    cmd("# IMI: adaptive rw (auto-detected TEMPORAL -> 0.15)")
    imi = space.navigate("recent auth issues", top_k=2)
    for m in imi.memories[:2]:
        output(f"  [{m['score']:.2f}] {m['content'][:70]}")
    print()

    output("\033[1;33mIMI boosts recent, actionable memories.\033[0m")
    output("\033[1;33mRAG treats all memories equally.\033[0m")
    time.sleep(1)

    # --- Summary ---
    section("Summary")
    output("84 tests | Zero LLM at query | SQLite only | MIT License")
    output("")
    output("\033[1mpip install imi-memory\033[0m")
    output("\033[1mhttps://github.com/RAG7782/imi\033[0m")
    print()


if __name__ == "__main__":
    main()
