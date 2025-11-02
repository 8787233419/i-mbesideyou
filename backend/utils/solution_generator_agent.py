# utils/solution_finder.py
import os
import pickle
from typing import Optional
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

EMBEDDING_PKL_PATH = os.getenv(
    "EMBEDDING_PKL_PATH",
    os.path.join("backend", "utils", "backend", "embeddings", "messi_embedding.pkl"),
)

DEFAULT_EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

_EMB: Optional[np.ndarray] = None
_TEXTS: Optional[list] = None
_MODEL_NAME: Optional[str] = None
_ST_MODEL = None


import os
import faiss
import pickle
import numpy as np

def _load_index():
    """
    Load the FAISS index and corresponding texts for retrieval.
    Returns: (index, texts)
    """
    # Paths relative to your current script
    base_dir = os.path.dirname(__file__)
    faiss_path = os.path.join(base_dir, "backend", "embeddings", "messi_index.faiss")
    texts_path = os.path.join(base_dir, "backend", "embeddings", "messi_texts.pkl")

    if not os.path.exists(faiss_path):
        raise FileNotFoundError(f"FAISS index not found at: {faiss_path}")
    if not os.path.exists(texts_path):
        raise FileNotFoundError(f"Texts file not found at: {texts_path}")

    # Load FAISS index
    index = faiss.read_index(faiss_path)

    # Load texts
    with open(texts_path, "rb") as f:
        texts = pickle.load(f)

    print("✅ FAISS index and texts loaded successfully.")
    return index, texts

def _retrieve_context(query: str, k: int = 3, max_chars: int = 12000) -> str:
    err = _load_index()
    if err:
        return err
    if not query.strip():
        return "Empty query."

    emb = _EMB
    texts = _TEXTS or []
    if emb is None or not len(texts):
        return "Embedding index not loaded."

    # Normalize matrix
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    emb_norm = emb / norms

    # Embed query locally
    try:
        q_vec = _ST_MODEL.encode([query], normalize_embeddings=True)[0]
    except Exception:
        return "Failed to embed query locally."

    sims = emb_norm @ q_vec.astype(np.float32)
    top_idx = np.argsort(-sims)[: max(1, k)]

    chunks = []
    total = 0
    for i in top_idx:
        t = str(texts[int(i)])
        if t and total < max_chars:
            remain = max_chars - total
            part = t[:remain]
            chunks.append(part)
            total += len(part)
        if total >= max_chars:
            break

    return "\n\n---\n\n".join(chunks) if chunks else "No relevant content found."


def find_solution(query: str):
    """Answer a query using learned Q&As first, then fallback to documentation search."""
    # Check learned Q&As first
    try:
        from sentence_transformers import SentenceTransformer
        from utils.unanswered_db import search_learned_qna
        
        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        query_embedding = query_vec.astype(np.float32).tobytes()
        
        learned_match = search_learned_qna(query_embedding, threshold=0.85)
        if learned_match:
            print(f"✅ Found learned answer (similarity: {learned_match['similarity']:.2%})")
            return f"[LEARNED] {learned_match['answer']}"
    except Exception as e:
        print(f"⚠️ Learned Q&A check failed: {e}")
    
    # Fallback to documentation search
    # Define a LangChain tool that searches the local embedding index
    docs_search_tool = Tool(
        name="docs_search",
        description="Search the project documentation by semantic similarity to the user's query.",
        func=lambda q: _retrieve_context(q, k=3),
    )

 
    try:
        llm_model = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(model=llm_model, temperature=0)
        agent = initialize_agent(
            tools=[docs_search_tool],
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            handle_parsing_errors=True,
        )
        prompt = (
            "Use the docs_search tool to find the most relevant solution to the user's query. "
            "Return only the most helpful solution of text as the final answer.\n\n"
            f"User query: {query}"
        )
        try:
            result = agent.invoke({"input": prompt})
        except Exception:
            result = agent.run(prompt)

        if isinstance(result, dict) and "output" in result:
            return result["output"]
        if isinstance(result, str):
            return result
        return str(result)
    except Exception:
        # Fallback to direct Gemini call with retrieved context (no tool)
        docs_text = _retrieve_context(query, k=3)
        model = genai.GenerativeModel(os.getenv("GENAI_MODEL", "gemini-2.5-flash"))
        prompt = f"""
        Use the docs_search tool to find the most relevant solution to the user's query. 
        Return only the most helpful solution of text as the final answer.\n\n
        Documentation:
        {docs_text}

        User query:
        {query}

        Reply concisely and only based on the documentation.
        """
        try:
            response = model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            return f"Error generating reply: {e}"


# if __name__ == "__main__":
#     print(find_solution("application is not working"))

