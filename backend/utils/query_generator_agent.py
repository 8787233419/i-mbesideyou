from __future__ import annotations
import os
import google.generativeai as genai

def generate_issue_statement(
    message: str,
    *,
    model: str = "gemini-2.5-flash",   
    temperature: float = 0.2,
    max_output_tokens: int = 128,
) -> str:
    """
    Analyzes a WhatsApp message and converts it into a structured query
    describing the actual issue for documentation search.
    """

    message = (message or "").strip()
    if not message:
        return "No message provided to analyze."

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "GOOGLE_API_KEY not set; cannot analyze the message."

    genai.configure(api_key=api_key)
    model_client = genai.GenerativeModel(model)

    instruction = """
You are a query generator for the Mess-i app — a hostel meal management system.
Your task is to analyze informal WhatsApp messages from users and determine
what specific issue or bug they are reporting.

Return a **concise single-sentence query** that clearly expresses
the technical issue in a way suitable for searching documentation or logs.

Do not include explanations, greetings, or any other text.
If the issue cannot be determined, reply with:
"unclear issue — needs human review."
"""

    prompt = (
        f"{instruction}\n\n"
        f"User Message:\n{message}\n\n"
        "Output: One short sentence describing the issue."
    )

    try:
        resp = model_client.generate_content(
            prompt,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_output_tokens,
            },
            safety_settings={
                "HARASSMENT": "BLOCK_NONE",
                "HATE_SPEECH": "BLOCK_NONE",
                "SEXUAL": "BLOCK_NONE",
                "DANGEROUS": "BLOCK_NONE",
            },
        )

        # ---- Safe text extraction ----
        text = None
        try:
            text = getattr(resp, "text", None)
            if text:
                text = text.strip()
        except Exception:
            text = None

        if not text and hasattr(resp, "candidates") and resp.candidates:
            cand = resp.candidates[0]
            if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                collected = []
                for p in cand.content.parts:
                    if hasattr(p, "text") and p.text:
                        collected.append(p.text)
                text = "\n".join(collected).strip() if collected else None

        return text or "unclear issue — needs human review."

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return f"Failed to analyze the message due to an API error: {e}"
