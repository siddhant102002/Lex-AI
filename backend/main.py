# main.py
# FastAPI backend for the Legal AI system
# Exposes two endpoints: /analyze for full contract analysis and /chat for Q&A
# The agent in legal_agent.py handles all AI logic — this file just handles requests

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import tempfile
import os
import sys

sys.path.append(os.path.dirname(__file__))

from document_reader import extract_text
from agents.legal_agent import run_agent

app = FastAPI(
    title="Legal AI API",
    description="Agentic legal document analysis powered by Claude",
    version="2.0"
)

# Required for React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    contract_text: str
    question: str

class AnalysisResponse(BaseModel):
    contract_type: str
    answer: str
    tools_used: list[str]

class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]


@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_contract(file: UploadFile = File(...)):
    """
    Accepts a PDF or DOCX contract file.
    Extracts the text, passes it to the legal agent, and returns a full analysis.
    The agent decides which tools to use based on the request.
    """
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX supported")

    suffix = "pdf" if file.filename.lower().endswith('.pdf') else "docx"
    temp_path = None

    try:
        # document_reader requires a file path, not raw bytes — so we save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
            content = await file.read()
            tmp.write(content)
            temp_path = tmp.name

        text = extract_text(temp_path)
        text = " ".join(text.split())

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read file: {e}")

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)

    if len(text) < 50:
        raise HTTPException(status_code=422, detail="Not enough text in document")

    question = """Please provide a complete analysis covering:
1. A plain English summary
2. All key clauses
3. Risk flags
4. Questions to ask before signing"""

    try:
        result = run_agent(question, text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")

    return AnalysisResponse(
        contract_type=_detect_type(text),
        answer=result["answer"],
        tools_used=result["tools_used"]
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_contract(request: ChatRequest):
    """
    Accepts a contract text and a question from the user.
    The agent selects the appropriate tool and returns a targeted answer.
    """
    if len(request.contract_text) < 50:
        raise HTTPException(status_code=422, detail="Contract text too short")

    try:
        result = run_agent(request.question, request.contract_text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    return ChatResponse(
        answer=result["answer"],
        tools_used=result["tools_used"]
    )


def _detect_type(text: str) -> str:
    """Basic keyword-based contract type detection."""
    text_lower = text[:2000].lower()
    if "non-disclosure" in text_lower or "nda" in text_lower:
        return "Non-Disclosure Agreement"
    elif "employment" in text_lower or "employee" in text_lower:
        return "Employment Contract"
    elif "service" in text_lower or "consultant" in text_lower:
        return "Service Agreement"
    elif "lease" in text_lower or "tenancy" in text_lower:
        return "Lease Agreement"
    else:
        return "Legal Contract"