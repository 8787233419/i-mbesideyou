from __future__ import annotations
from typing import Optional
import os

def generate_final_reply(
    query: str,
    results: list[str],
    *,
    model: str = "gemini-2.5-flash",
    temperature: float = 0.2,
    max_output_tokens: int = 512,
) -> dict:
    query = (query or "").strip()
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return {"reply": "GOOGLE_API_KEY not set; cannot generate reply.", "confidence": 0}

    try:
        import google.generativeai as genai
    except Exception:
        return {"reply": "google-generativeai SDK not available; cannot generate reply.", "confidence": 0}

    try:
        genai.configure(api_key=api_key)
        model_client = genai.GenerativeModel(model)

        # instruction = """
            
        #     You are a reply generator assistant, where you will be given a query formatted out of whatsapp text message and a few results or facts from the solution finder tool regarding any issue in the project mess-i 
        #     (an app that digitses meal taking process in hostel wiht other multiple features like rebates, id card tapping, since there are multiple features there can be some errors)
        #     your task is to analyse the query and the results or facts and generate a reply to the user same as the sender would have sent the message.
        #     Return a reply to the user that will be helpful to the user to and would be same if the sender would have sent the message as you are trying to mimic the sender's message.
        #     Do not include any other text in your response.
        #     IMPORTANT:
        #         You are not allowed to hallucinate any solution by your own unless you feel that issue can't be solved based on documentation in that case reply should be its a server error we will get shortly back to you and your reply should be in the same language as the query.
            
        # """

        instruction = """
You are a WhatsApp support assistant for the **Mess-i app**, which manages hostel meal systems,
rebates, and ID card scanning.

You receive user messages about errors or issues they face in the app.

You will be given:
1. The **user's WhatsApp message** (usually short, casual, sometimes with grammar mistakes).
2. A list of **facts or results** found in documentation related to the issue.

Your task:
- Understand the user's problem.
- Use the facts to craft a *human-like, friendly WhatsApp reply* as if you are a **Mess-i support team member**.
- If the problem is caused by **user role limits** (e.g., only managers can approve rebates), politely explain that.
- If documentation doesn't help, reply with:
  "It seems like a temporary server issue. We'll fix it soon. If it continues, please contact the leads personally."

Tone guidelines:
- Keep it short (1–3 sentences)
- No greetings unless necessary
- Use the same language and tone as the user's query (formal/informal)
- Never invent new features or instructions not found in the facts

**IMPORTANT OUTPUT FORMAT:**
Return your response as a JSON object with this exact format:
{"reply": "your message to the user", "confidence": <number 0-100>}

The confidence score should reflect:
- 60-100: Strong documentation support, specific solution
- 40-59: Moderate support, some uncertainty
- 20-39: Generic/fallback response
- 0-19: Unable to help, needs human review
"""


        prompt = (
            f"{instruction}\n\n"
            f"Whatsapp Message:\n{query}\n\n"
            f"Results or Facts:\n{results}\n\n"
            "Output: JSON object with reply and confidence score."
        )

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

        # Handle incomplete responses
        if resp.candidates and hasattr(resp.candidates[0], "finish_reason"):
            reason = resp.candidates[0].finish_reason
            # finish_reason: 1=STOP (normal), 2=MAX_TOKENS (truncated), 3=SAFETY, 4=RECITATION, 5=OTHER
            if reason == 3:
                return {"reply": "Response blocked by safety filters.", "confidence": 20}
            elif reason == 2:
                # Truncated due to max tokens - mark as low confidence
                print("⚠️ Response was truncated due to max_output_tokens")


        # Extract text safely
        # ---- Extract text safely ----
        text = ""

        try:
            # Try the quick accessor
            text = getattr(resp, "text", None)
            if text:
                text = text.strip()
        except Exception:
            text = None

        # If that failed, fallback to deeper structure
        if not text and hasattr(resp, "candidates") and resp.candidates:
            cand = resp.candidates[0]
            if hasattr(cand, "content") and hasattr(cand.content, "parts"):
                parts = cand.content.parts
                if parts:
                    collected = []
                    for p in parts:
                        if hasattr(p, "text") and p.text:
                            collected.append(p.text)
                    text = "\n".join(collected).strip()

        # Final fallback
        if not text:
            return {
                "reply": "Could not generate a reply — empty response or truncated output.",
                "confidence": 30
            }

        # Check if using learned answer (high confidence)
        if results.startswith("[LEARNED]"):
            actual_answer = results.replace("[LEARNED]", "").strip()
            return {
                "reply": actual_answer,
                "confidence": 95
            }
        
        # Parse JSON response
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{[^}]*"reply"[^}]*\}', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                result = json.loads(json_str)
                reply = result.get("reply", text)
                confidence = float(result.get("confidence", 50))
                confidence = max(0.0, min(100.0, confidence))
                
                return {
                    "reply": reply,
                    "confidence": confidence
                }
            except (json.JSONDecodeError, ValueError):
                pass
        
        # Fallback: treat as plain text, estimate confidence
        confidence = 50
        text_lower = text.lower()
        
        # Check if response seems incomplete/truncated
        is_truncated = False
        if resp.candidates and hasattr(resp.candidates[0], "finish_reason"):
            if resp.candidates[0].finish_reason == 2:  # MAX_TOKENS
                is_truncated = True
        
        # Also check for incomplete sentences
        if text and not any(text.rstrip().endswith(char) for char in ['.', '!', '?', '।', '।।']):
            # Doesn't end with punctuation - likely truncated
            is_truncated = True
        
        if is_truncated:
            print("⚠️ Response appears truncated or incomplete")
            confidence = 25  # Low confidence for truncated responses
        else:
            # Rule-based confidence estimation
            if any(phrase in text_lower for phrase in [
                "server issue", "temporary issue", "contact the leads",
                "we'll fix it soon", "unclear", "unable to"
            ]):
                confidence = 35
            elif any(phrase in text_lower for phrase in [
                "you can", "try", "should be able to", "make sure"
            ]):
                confidence = 75
        
        return {
            "reply": text,
            "confidence": confidence
        }


    except Exception as e:
        import traceback
        print(f"❌ Reply generation error: {e}")
        traceback.print_exc()
        return {
            "reply": f"Failed to generate final reply due to an API error: {str(e)}",
            "confidence": 10
        }
