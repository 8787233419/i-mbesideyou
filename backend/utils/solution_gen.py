import os
import time
import pickle
import numpy as np
import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

DOC_PATH = os.getenv("DOC_PATH", "backend/docs/messi_doc.txt")
with open(DOC_PATH, "r", encoding="utf-8") as f:
    text = f.read()

def chunk_text(text, max_chars=1500):
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]

chunks = chunk_text(text)
model = "text-embedding-004"

embeddings = []
for i, chunk in enumerate(chunks):
    try:
        result = genai.embed_content(model=model, content=chunk)
        embeddings.append(result["embedding"])
        print(f"✅ Chunk {i+1}/{len(chunks)} embedded successfully.")
        time.sleep(1)
    except Exception as e:
        print(f"⚠️ Error embedding chunk {i+1}: {e}")
        time.sleep(10)

# Save both embeddings and texts
os.makedirs("backend/utils/backend/embeddings", exist_ok=True)
data = {
    "embeddings": np.array(embeddings, dtype=np.float32),
    "texts": chunks,
    "model": model
}

with open("backend/utils/backend/embeddings/messi_embedding.pkl", "wb") as f:
    pickle.dump(data, f)

print("\n✅ Embeddings + texts saved successfully at backend/utils/backend/embeddings/messi_embedding.pkl")



# # utils/solution_finder.py
# import google.generativeai as genai
# import os
# from dotenv import load_dotenv

# load_dotenv()

# genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
# DOC_PATH = os.getenv("DOC_PATH", "backend/docs/messi_doc.txt")

# def _read_docs(max_chars: int = 15000) -> str:
#     if not os.path.exists(DOC_PATH):
#         return f"Documentation not found at {DOC_PATH}"
#     try:
#         with open(DOC_PATH, "r", encoding="utf-8", errors="ignore") as f:
#             text = f.read()
#         return text[:int(max_chars)]
#     except Exception as e:
#         return f"Error reading documentation: {e}"

# def find_solution(query: str):
#     """Answer a query using Gemini directly over the documentation text."""
#     docs_text = _read_docs()
#     model = genai.GenerativeModel("gemini-2.5-flash")  # Or "gemini-1.5-pro"

#     prompt = f"""
#     Use the docs_search tool to find the most relevant solution to the user's query. 
#     Return only the most helpful solution of text as the final answer.\n\n

#     Documentation:
#     {docs_text}

#     User query:
#     {query}

#     Reply concisely and only based on the documentation.
#     """

#     try:
#         response = model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         return f"Error generating reply: {e}"

# if __name__ == "__main__":
#     print(find_solution("I am a student not able to accept rebates of students"))
#     # genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
#     # for m in genai.list_models():
#     #     print(m.name) 
