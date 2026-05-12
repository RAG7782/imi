"""Tests for the LangChain integration."""

import os

# Force OllamaLLM backend so tests don't require the 'anthropic' package
os.environ.setdefault("IMI_LLM_BACKEND", "ollama")

from imi.integrations.langchain import IMIMemory


class TestIMIMemory:
    def test_memory_variables(self):
        mem = IMIMemory()
        assert mem.memory_variables == ["history"]

    def test_save_and_load(self):
        mem = IMIMemory()
        mem.save_context(
            {"input": "DNS failure at 03:00"},
            {"output": "Restarting DNS and checking cascade"},
        )
        result = mem.load_memory_variables({"input": "DNS issues"})
        assert "history" in result
        assert len(result["history"]) > 0
        # Should find the DNS memory
        assert "DNS" in result["history"] or "dns" in result["history"].lower()

    def test_multiple_memories(self):
        mem = IMIMemory()
        mem.save_context(
            {"input": "Auth service down"},
            {"output": "Checking upstream dependencies"},
        )
        mem.save_context(
            {"input": "Database timeout"},
            {"output": "Connection pool exhausted, scaling up"},
        )
        result = mem.load_memory_variables({"input": "database connection"})
        assert "history" in result
        assert len(result["history"]) > 0

    def test_clear(self):
        mem = IMIMemory()
        mem.save_context({"input": "test"}, {"output": "response"})
        assert len(mem.space.episodic) > 0
        mem.clear()
        assert len(mem.space.episodic) == 0

    def test_empty_query(self):
        mem = IMIMemory()
        result = mem.load_memory_variables({"input": ""})
        assert result["history"] == ""

    def test_custom_keys(self):
        mem = IMIMemory(memory_key="context", input_key="question", output_key="answer")
        mem.save_context(
            {"question": "What happened?"},
            {"answer": "DNS failure"},
        )
        result = mem.load_memory_variables({"question": "DNS"})
        assert "context" in result

    def test_direct_encode(self):
        mem = IMIMemory()
        node = mem.encode("Direct memory encoding test", tags=["test"])
        assert node.id
        assert "test" in node.tags

    def test_dream(self):
        mem = IMIMemory()
        report = mem.dream()
        assert report.nodes_processed == 0  # empty space

    def test_search_actions(self):
        mem = IMIMemory()
        results = mem.search_actions("restart")
        assert isinstance(results, list)

    def test_from_sqlite(self, tmp_path):
        db = str(tmp_path / "test.db")
        mem = IMIMemory.from_sqlite(db)
        mem.save_context({"input": "test"}, {"output": "ok"})
        assert len(mem.space.episodic) == 1

        # Reload — memories should persist
        mem2 = IMIMemory.from_sqlite(db)
        assert len(mem2.space.episodic) == 1
