"""IMI integration with LangChain.

Provides IMIMemory — a LangChain-compatible memory backend that uses IMI
for cognitive memory (temporal decay, affordances, graph-augmented retrieval).

Two usage patterns:

1. Standalone (no LangChain dependency required):
    ```python
    from imi.integrations.langchain import IMIMemory
    memory = IMIMemory.from_sqlite("agent.db")
    memory.save_context({"input": "DNS failed"}, {"output": "Restarting..."})
    relevant = memory.load_memory_variables({"input": "auth issues"})
    ```

2. With LangChain chain:
    ```python
    from langchain.chains import ConversationChain
    from imi.integrations.langchain import IMIMemory
    memory = IMIMemory.from_sqlite("agent.db")
    chain = ConversationChain(llm=llm, memory=memory)
    ```
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


class IMIMemory:
    """LangChain-compatible memory backed by IMI.

    Implements the memory interface expected by LangChain:
      - load_memory_variables(inputs) → dict with memory key
      - save_context(inputs, outputs) → None
      - clear() → None

    Also exposes IMI-specific features:
      - navigate(query) → NavigationResult
      - search_actions(query) → list of affordances
      - dream() → consolidation report
    """

    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "output"
    return_messages: bool = False

    def __init__(
        self,
        space: Any = None,
        *,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        top_k: int = 5,
        zoom: str = "medium",
        include_affordances: bool = True,
    ):
        """Initialize IMI memory.

        Args:
            space: An IMISpace instance. If None, creates a new one.
            memory_key: Key used to return memory context to the chain.
            input_key: Key for user input in the inputs dict.
            output_key: Key for AI output in the outputs dict.
            top_k: Number of memories to retrieve per query.
            zoom: Resolution level for retrieved memories.
            include_affordances: Whether to include affordances in context.
        """
        if space is None:
            from imi.space import IMISpace

            space = IMISpace()
        self._space = space
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self._top_k = top_k
        self._zoom = zoom
        self._include_affordances = include_affordances

    @classmethod
    def from_sqlite(cls, db_path: str, **kwargs) -> IMIMemory:
        """Create IMIMemory backed by SQLite persistence.

        M12 fix: only use JSON load if persist_dir exists AND is a directory.
        A file with the same stem (e.g., 'agent' for 'agent.db') would
        incorrectly trigger the JSON path.
        """
        from imi.space import IMISpace

        persist_dir = Path(db_path).with_suffix("")
        if persist_dir.exists() and persist_dir.is_dir():
            space = IMISpace.load(persist_dir)
        else:
            space = IMISpace.from_sqlite(db_path)
        return cls(space=space, **kwargs)

    @property
    def memory_variables(self) -> list[str]:
        """Return memory variables (LangChain interface)."""
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Retrieve relevant memories for the current input.

        Uses IMI's adaptive relevance weighting and graph expansion.
        """
        query = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: ""}

        nav = self._space.navigate(
            query,
            zoom=self._zoom,
            top_k=self._top_k,
        )

        lines = []
        for m in nav.memories:
            content = m["content"]
            score = m["score"]
            lines.append(f"[{score:.2f}] {content}")
            if self._include_affordances and m.get("affordances"):
                for aff in m["affordances"][:2]:
                    lines.append(f"  -> {aff}")

        context = "\n".join(lines) if lines else "No relevant memories found."

        if self.return_messages:
            # For chat models, return as system message content
            return {self.memory_key: context}
        return {self.memory_key: context}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        """Save the interaction as a memory.

        Encodes both the input and output as a single memory node.
        """
        user_input = inputs.get(self.input_key, "")
        ai_output = outputs.get(self.output_key, "")

        experience = f"User: {user_input}\nAssistant: {ai_output}"
        self._space.encode(
            experience,
            source="langchain",
            tags=["conversation"],
        )

    def clear(self) -> None:
        """Clear all memories (creates fresh stores)."""
        from imi.store import VectorStore

        self._space.episodic = VectorStore()
        self._space.semantic = VectorStore()
        if self._space.persist_dir:
            self._space.save()

    # ---- IMI-specific methods ----

    def navigate(self, query: str, **kwargs) -> Any:
        """Direct access to IMI's navigate for advanced usage."""
        return self._space.navigate(query, **kwargs)

    def search_actions(self, query: str, top_k: int = 5) -> list[dict]:
        """Search memories by what actions they enable."""
        return self._space.search_affordances(query, top_k=top_k)

    def dream(self) -> Any:
        """Run consolidation cycle."""
        return self._space.dream()

    def encode(self, experience: str, **kwargs) -> Any:
        """Direct access to IMI's encode."""
        return self._space.encode(experience, **kwargs)

    @property
    def space(self) -> Any:
        """Access the underlying IMISpace."""
        return self._space
