# LangChain Integration

**Coming soon.**

A native LangChain integration for IMI is planned. The target interface will expose IMI as a LangChain `BaseRetriever` and `BaseMemory` compatible component.

## Planned interface

```python
# Planned — not yet implemented

from imi.integrations.langchain import IMIRetriever, IMIMemory

# As a retriever
retriever = IMIRetriever.from_sqlite(
    db_path="agent.db",
    top_k=10,
    zoom="medium",
)

# Use in a chain
chain = RetrievalQA.from_chain_type(
    llm=ChatAnthropic(),
    retriever=retriever,
)

# As a memory component
memory = IMIMemory.from_sqlite(
    db_path="agent.db",
    return_messages=True,
)

agent = ConversationChain(llm=ChatAnthropic(), memory=memory)
```

## Workaround

Until the native integration ships, you can wrap IMI directly using the REST API:

```python
from langchain.schema import BaseRetriever, Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
import httpx

class IMIRetriever(BaseRetriever):
    base_url: str = "http://localhost:8000"
    top_k: int = 10
    zoom: str = "medium"

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        resp = httpx.post(
            f"{self.base_url}/navigate",
            json={"query": query, "top_k": self.top_k, "zoom": self.zoom},
        )
        resp.raise_for_status()
        data = resp.json()
        return [
            Document(
                page_content=m["content"],
                metadata={"score": m["score"], "id": m["id"], "tags": m["tags"]},
            )
            for m in data["memories"]
        ]
```

Start the REST API first (`uvicorn imi.api:app --port 8000`), then use the retriever in any LangChain chain or agent.

## Track progress

Watch the [GitHub repository](https://github.com/RAG7782/imi) for updates on the LangChain integration.
