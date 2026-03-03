"""
Streamlit Chat Interface for the RAG Chatbot.

This is a starter template — feel free to modify, extend, or replace it entirely.
Run with: streamlit run src/app.py
"""

from dotenv import load_dotenv
load_dotenv()
import streamlit as st
from langchain_core.globals import set_llm_cache
from langchain_community.cache import InMemoryCache
from graph import run_graph
from pipeline import rag_chain
set_llm_cache(InMemoryCache()) # cache repeated queries for faster responses

# ──────────────────────────────────────────────
# UI Components
# ──────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render sidebar settings. Returns a dict of user-configured parameters."""
    with st.sidebar:
        st.header("⚙️ Settings")

        top_k = st.slider(
            "Retrieved chunks (Top-K)",
            min_value=1,
            max_value=10,
            value=3,
            help="How many document chunks to retrieve per query.",
        )

        st.divider()
        st.markdown("**How it works**")
        st.markdown(
            "1. Your question is embedded using **nomic-embed-text**\n"
            "2. **LangGraph** retrieves relevant chunks via **MMR** search\n"
            "3. **llama3.2:3b** generates a grounded answer from context"
        )

        st.divider()
        # clear chat history
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

    return {"top_k": top_k}


def render_message(message: dict) -> None:
    """Render a single chat message with optional source expander."""
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")


def render_chat_history() -> None:
    """Display all messages stored in session state."""
    for message in st.session_state.messages:
        render_message(message)


def get_bot_response(query: str, top_k: int) -> tuple[str, list[str]]:
    # LangGraph handles retrieval — generation done separately for streaming support
    context, sources = run_graph(query, top_k)
    return context, sources

# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="Customer Service Chatbot",
        page_icon="🚗",

        layout="centered",
    )

    st.title("🚗 Customer Service Chatbot")
    st.caption("Powered by llama3.2:3b · ChromaDB · LangChain · LangGraph")
    st.image("assets/Gemini_Generated_Image_frzkn1frzkn1frzk.png", use_container_width=True)
    st.divider()

    # Sidebar
    settings = render_sidebar()

    # Session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        st.info("👋 Ask me anything about vehicles, warranty, service, or ordering process.")

    render_chat_history()

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        # User message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="🧍‍♂️"):
            st.markdown(prompt)

        # Bot response
        with st.status("Processing your query...", expanded=True) as status:
            st.write("🔍 Retrieving relevant chunks from ChromaDB...")
            context, sources = get_bot_response(prompt, top_k=settings["top_k"])
            st.write("🤖 Generating response...")
            status.update(label="✅ Done!", state="complete", expanded=False)

        with st.chat_message("assistant", avatar="🤖"):
            response_placeholder = st.empty()
            full_response = ""
            for chunk in rag_chain.stream({"context": context, "question": prompt}, # stream LLM response token by token — better UX than waiting for full response
            config={
            "tags": ["llama3.2:3b-mmr-chunk600"],  # Metadata for the langsmith tracking
            "metadata": {
                "model": "llama3.2:3b",
                "retriever": "mmr",
                "chunk_size":600,
                "num_predict":200
            }
             }
                ):
                full_response += chunk
                response_placeholder.markdown(full_response + "▌")
            response_placeholder.markdown(full_response)
            st.toast("Response generated!", icon="✅")
        if sources:
            with st.expander("📄 Sources"):
                for source in sources:
                    st.markdown(f"- `{source}`") 

        st.session_state.messages.append({
        "role": "assistant",
        "content": full_response,
        "sources": sources
        })

if __name__ == "__main__":
    main()
