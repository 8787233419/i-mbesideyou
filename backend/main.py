from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import uvicorn
from utils.reply_generator import generate_final_reply
from utils.solution_generator_agent import find_solution
from utils.query_generator_agent import generate_issue_statement
from utils.unanswered_db import (
    save_low_confidence_query,
    get_pending_queries,
    get_all_queries,
    get_query_stats,
    update_query_status,
    delete_query,
    save_learned_qna
)

app = FastAPI(title="Mess-i Support Agent")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/generate-reply")
async def generate_reply(request: Request):
    """
    Generate a reply to a user's WhatsApp message.
    Automatically tracks low-confidence responses.
    """
    try:
        body = await request.json()
    except Exception:
        body = None

    message = None
    if isinstance(body, dict):
        message = body.get("message")
    elif isinstance(body, str):
        message = body

    if not message:
        return {"error": "No message provided", "reply": ""}

    # Step 1: Generate structured query
    query = generate_issue_statement(message)
    print(f"📝 Query: {query}")
    
    # Step 2: Find solution from documentation
    results = find_solution(query)
    print(f"🔍 Results: {results[:200]}...")
    
    # Step 3: Generate final reply with confidence (single LLM call)
    reply_data = generate_final_reply(query, results)
    
    # Handle response format
    if isinstance(reply_data, dict):
        reply = reply_data.get("reply", "")
        confidence = reply_data.get("confidence", 50)
    else:
        # Fallback if old format is returned
        reply = str(reply_data)
        confidence = 50
    
    print(f"💬 Reply: {reply}")
    print(f"🎯 Confidence: {confidence}%")
    
    # Step 4: Save to DB if confidence is low or if reply generation failed
    saved_to_db = False
    CONFIDENCE_THRESHOLD = 70.0
    original_reply = reply  # Keep original for database
    
    # Check if this is a learned answer (high confidence from DB)
    is_learned_answer = confidence >= 90
    
    # Check if reply generation failed or is incomplete
    reply_lower = reply.lower()
    reply_failed = any(phrase in reply_lower for phrase in [
        "could not generate",
        "failed to generate",
        "empty response",
        "api error",
        "api key not set",
        "blocked by safety"
    ])
    
    # Check if reply seems truncated - but SKIP this check for learned answers
    is_truncated = False
    if not is_learned_answer:
        is_truncated = reply and len(reply) > 20 and not any(
            reply.rstrip().endswith(char) for char in ['.', '!', '?', '।', '।।', '"', "'"]
        )
        
        if is_truncated:
            print(f"⚠️ Reply appears truncated: '{reply[-50:]}'")
            reply_failed = True
    
    # Save to DB if confidence is low OR reply generation failed
    if confidence <= CONFIDENCE_THRESHOLD or reply_failed:
        # Replace uncertain/failed reply with standard fallback message
        reply = "Thanks for reaching out! We've noted your issue and our team will get back to you shortly. If it's urgent, please contact the leads personally."
        
        reason = "Reply generation failed" if reply_failed else f"Confidence {confidence}% (threshold: {CONFIDENCE_THRESHOLD}%)"
        
        try:
            query_id = save_low_confidence_query(
                original_message=message,
                generated_query=query,
                attempted_solution=results,
                generated_reply=original_reply,  # Save original reply for review
                confidence_score=confidence,
                reason=reason
            )
            saved_to_db = True
            print(f"⚠️ {reason}. Saved to DB (ID: {query_id})")
        except Exception as e:
            print(f"❌ Failed to save to DB: {e}")
    
    return {
        "reply": reply,
        "confidence": confidence,
        "needs_review": saved_to_db,
        "query": query
    }


@app.get("/unanswered-queries")
async def get_unanswered_queries(
    status: Optional[str] = "pending",
    limit: int = 50,
    offset: int = 0
):
    """
    Get all unanswered/low-confidence queries.
    
    Query params:
    - status: Filter by status ('pending', 'resolved', 'dismissed', or None for all)
    - limit: Maximum number of results (default: 50)
    - offset: Pagination offset (default: 0)
    """
    try:
        if status:
            queries = get_all_queries(status=status, limit=limit, offset=offset)
        else:
            queries = get_all_queries(limit=limit, offset=offset)
        
        return {
            "success": True,
            "count": len(queries),
            "queries": queries
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/query-stats")
async def get_stats():
    """
    Get statistics about unanswered queries.
    """
    try:
        stats = get_query_stats()
        return {
            "success": True,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/unanswered-queries/{query_id}")
async def update_query(query_id: int, request: Request):
    """
    Update a query's status or add human response.
    
    Body params:
    - status: New status ('resolved', 'dismissed', 'pending')
    - human_response: Optional human response to the query
    """
    try:
        body = await request.json()
        status = body.get("status")
        human_response = body.get("human_response")
        
        if not status:
            raise HTTPException(status_code=400, detail="Status is required")
        
        if status not in ["pending", "resolved", "dismissed"]:
            raise HTTPException(
                status_code=400,
                detail="Status must be 'pending', 'resolved', or 'dismissed'"
            )
        
        success = update_query_status(query_id, status, human_response)
        
        if not success:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return {
            "success": True,
            "message": f"Query {query_id} updated to status: {status}"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/unanswered-queries/{query_id}")
async def remove_query(query_id: int):
    """
    Delete a query from the database.
    """
    try:
        success = delete_query(query_id)
        
        if not success:
            raise HTTPException(status_code=404, detail="Query not found")
        
        return {
            "success": True,
            "message": f"Query {query_id} deleted successfully"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/approve-answer")
async def approve_answer(request: Request):
    """
    Save admin's approved answer as learned Q&A.
    Body: {query_id, query, approved_answer}
    """
    try:
        body = await request.json()
        query_id = body.get("query_id")
        query = body.get("query")
        approved_answer = body.get("approved_answer")
        
        if not query or not approved_answer:
            raise HTTPException(status_code=400, detail="query and approved_answer required")
        
        # Generate embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        query_vec = model.encode([query], normalize_embeddings=True)[0]
        query_embedding = query_vec.astype('float32').tobytes()
        
        # Save to learned Q&A
        qna_id = save_learned_qna(query, approved_answer, query_embedding)
        
        # Update original query status if query_id provided
        if query_id:
            update_query_status(query_id, "resolved", approved_answer)
        
        return {
            "success": True,
            "qna_id": qna_id,
            "message": "Answer saved and will be used for similar queries"
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """
    API health check and info endpoint.
    """
    stats = get_query_stats()
    return {
        "status": "running",
        "service": "Mess-i Support Agent",
        "version": "2.0",
        "endpoints": {
            "generate_reply": "/generate-reply (POST)",
            "unanswered_queries": "/unanswered-queries (GET)",
            "query_stats": "/query-stats (GET)",
            "update_query": "/unanswered-queries/{id} (PUT)",
            "delete_query": "/unanswered-queries/{id} (DELETE)",
            "approve_answer": "/approve-answer (POST)"
        },
        "current_stats": stats
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)