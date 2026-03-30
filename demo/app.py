"""IMI Live Demo — Gradio app for Hugging Face Spaces.

Run locally:
    pip install gradio
    python demo/app.py

Deploy to HF Spaces:
    Copy demo/ to a HF Space repo with requirements.txt
"""

import json
import gradio as gr

# Lazy-load IMI to avoid slow startup in UI
_space = None

def get_space():
    global _space
    if _space is None:
        from imi.space import IMISpace
        _space = IMISpace.from_sqlite("/tmp/imi_demo_gradio.db")

        # Pre-load sample incidents if empty
        if len(_space.episodic) == 0:
            samples = [
                ("DNS resolution failure at 03:00 UTC caused authentication service cascade failure across 3 microservices", ["dns", "auth", "incident"]),
                ("Auth service recovered after DNS fix at 03:45, but 200 users lost active sessions", ["auth", "recovery"]),
                ("Database connection pool exhausted during peak load, causing 500 errors on /api/users", ["database", "performance"]),
                ("Deployed circuit breaker pattern on auth service to prevent future cascade failures", ["auth", "improvement"]),
                ("SSL certificate for api.example.com expired, causing HTTPS handshake failures", ["ssl", "incident"]),
                ("Kubernetes pod OOMKilled: memory leak in image processing service consuming 4GB", ["kubernetes", "memory"]),
                ("Rolling deployment of v2.3 caused 5-minute downtime due to incompatible API change", ["deployment", "incident"]),
                ("Redis sentinel failover during network partition, 30 seconds of cache misses", ["redis", "incident"]),
                ("Automated SSL cert renewal with Let's Encrypt preventing future expiry incidents", ["ssl", "improvement"]),
                ("Post-mortem: DNS failure root cause was misconfigured TTL after provider migration", ["dns", "postmortem"]),
            ]
            for text, tags in samples:
                _space.encode(text, tags=tags)
    return _space


def encode_memory(experience: str, tags: str):
    """Encode a new memory."""
    space = get_space()
    tag_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
    node = space.encode(experience, tags=tag_list)

    affect = f"salience={node.affect.salience:.2f}, valence={node.affect.valence:.2f}" if node.affect else "none"
    affordances = "\n".join([f"  [{a.confidence:.0%}] {a.action}" for a in node.affordances[:3]]) or "  (none — needs LLM)"

    return f"""**Memory encoded successfully**

**ID:** `{node.id}`
**Summary:** {node.summary_medium}
**Tags:** {', '.join(node.tags)}
**Affect:** {affect}
**Mass:** {node.mass:.3f}
**Affordances:**
{affordances}

**Total memories:** {len(space.episodic)} episodic, {len(space.semantic)} semantic
"""


def navigate_memory(query: str, top_k: int, zoom: str):
    """Navigate the memory space."""
    space = get_space()
    rw, intent = space.adaptive_rw.classify_with_info(query)

    result = space.navigate(query, zoom=zoom, top_k=int(top_k))

    lines = [
        f"**Query:** {query}",
        f"**Intent detected:** {intent.name} -> rw={rw}",
        f"**Zoom:** {zoom}",
        f"**Hits:** {len(result.memories)}",
        "",
    ]

    for i, m in enumerate(result.memories[:int(top_k)]):
        lines.append(f"### Result {i+1} (score: {m['score']:.3f})")
        lines.append(f"{m['content']}")
        if m.get('tags'):
            lines.append(f"*Tags: {', '.join(m['tags'])}*")
        if m.get('affordances'):
            lines.append(f"*Affordances: {', '.join(m['affordances'][:2])}*")
        lines.append("")

    return "\n".join(lines)


def search_actions(action_query: str, top_k: int):
    """Search by affordances."""
    space = get_space()
    results = space.search_affordances(action_query, top_k=int(top_k))

    if not results:
        return "No affordances found. Encode memories with LLM enabled (`pip install imi-memory[llm]`) to generate affordances."

    lines = [f"**Actions for:** \"{action_query}\"\n"]
    for r in results:
        lines.append(f"- **[{r['confidence']:.0%}]** {r['action']}")
        lines.append(f"  *Conditions:* {r['conditions']}")
        lines.append(f"  *From:* {r['memory_summary'][:80]}...")
        lines.append("")

    return "\n".join(lines)


def get_stats():
    """Get memory space statistics."""
    space = get_space()
    graph = space.graph.stats()

    return f"""**Memory Space Statistics**

| Metric | Value |
|--------|-------|
| Episodic memories | {len(space.episodic)} |
| Semantic memories | {len(space.semantic)} |
| Total | {len(space.episodic) + len(space.semantic)} |
| Graph edges | {graph['total_edges']} |
| Annealing iteration | {space.annealing.iteration} |
| Converged | {space.annealing.converged} |
"""


def run_dream():
    """Run consolidation cycle."""
    space = get_space()
    report = space.dream()

    return f"""**Dream cycle complete**

| Metric | Value |
|--------|-------|
| Nodes processed | {report.nodes_processed} |
| Clusters formed | {report.clusters_formed} |
| Patterns | {report.patterns_total} |
| Episodic after | {len(space.episodic)} |
| Semantic after | {len(space.semantic)} |
"""


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(
    title="IMI — Integrated Memory Intelligence",
    theme=gr.themes.Soft(primary_hue="purple"),
) as demo:
    gr.Markdown("""
# IMI — Integrated Memory Intelligence

### *RAG finds what's similar. IMI finds what matters.*

Cognitive memory for AI agents: temporal decay, affordances, graph multi-hop, adaptive relevance weighting.
Zero LLM calls at query time.
    """)

    with gr.Tab("Encode"):
        gr.Markdown("### Store a new memory")
        encode_input = gr.Textbox(
            label="Experience",
            placeholder="DNS failure at 03:00 caused auth cascade across 3 services",
            lines=3,
        )
        encode_tags = gr.Textbox(
            label="Tags (comma-separated)",
            placeholder="dns, auth, incident",
        )
        encode_btn = gr.Button("Encode", variant="primary")
        encode_output = gr.Markdown()
        encode_btn.click(encode_memory, inputs=[encode_input, encode_tags], outputs=encode_output)

    with gr.Tab("Navigate"):
        gr.Markdown("### Search memories with adaptive relevance weighting")
        nav_query = gr.Textbox(
            label="Query",
            placeholder="recent auth failures",
        )
        with gr.Row():
            nav_top_k = gr.Slider(1, 20, value=5, step=1, label="Top K")
            nav_zoom = gr.Dropdown(
                ["orbital", "medium", "detailed", "full"],
                value="medium",
                label="Zoom Level",
            )
        nav_btn = gr.Button("Navigate", variant="primary")
        nav_output = gr.Markdown()
        nav_btn.click(navigate_memory, inputs=[nav_query, nav_top_k, nav_zoom], outputs=nav_output)

    with gr.Tab("Actions"):
        gr.Markdown("### Search by what you want to DO, not just content")
        action_query = gr.Textbox(
            label="Action query",
            placeholder="prevent outages",
        )
        action_top_k = gr.Slider(1, 10, value=5, step=1, label="Top K")
        action_btn = gr.Button("Search Actions", variant="primary")
        action_output = gr.Markdown()
        action_btn.click(search_actions, inputs=[action_query, action_top_k], outputs=action_output)

    with gr.Tab("Dream"):
        gr.Markdown("### Run consolidation (like sleep consolidation)")
        dream_btn = gr.Button("Dream", variant="primary")
        dream_output = gr.Markdown()
        dream_btn.click(run_dream, outputs=dream_output)

    with gr.Tab("Stats"):
        gr.Markdown("### Memory space statistics")
        stats_btn = gr.Button("Refresh Stats", variant="primary")
        stats_output = gr.Markdown()
        stats_btn.click(get_stats, outputs=stats_output)

    gr.Markdown("""
---
**[GitHub](https://github.com/RAG7782/imi)** · **[Paper](https://github.com/RAG7782/imi/blob/main/docs/arxiv/imi-paper.pdf)** · `pip install imi-memory` · MIT License
    """)


if __name__ == "__main__":
    demo.launch()
