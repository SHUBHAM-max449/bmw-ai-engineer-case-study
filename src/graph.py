"""
graph.py — LangGraph Pipeline
Handles retrieval node only.
Generation is handled separately in app.py for streaming support.
START -> retrieve_node -> END
"""

from pathlib import Path
from typing import TypedDict
from langgraph.graph import StateGraph, END, START
from ingestion import vectorstore

#  Graph State 

class GraphState(TypedDict):
    query   : str
    top_k   : int
    context : str
    sources : list[str]

#  Node 1: Retrieve 

def retrieve_node(state: GraphState) -> GraphState:
    query = state["query"]
    top_k = state.get("top_k", 3)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k}
    )
    docs = retriever.invoke(query)

    context = "\n\n".join(
        f"[Source: {Path(doc.metadata['source']).name}]\n{doc.page_content}"
        for doc in docs
    )
    sources = list({Path(doc.metadata["source"]).name for doc in docs})

    return {**state, "context": context, "sources": sources}

#  Build Graph 
workflow = StateGraph(GraphState)
workflow.add_node("retrieve", retrieve_node)

workflow.add_edge(START,"retrieve")
workflow.add_edge("retrieve", END)

graph = workflow.compile()

#  Public Interface 
def run_graph(query: str, top_k: int = 3) -> tuple[str, list[str]]:
    result = graph.invoke({"query": query, "top_k": top_k})
    return result["context"], result["sources"]