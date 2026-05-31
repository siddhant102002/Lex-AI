# main.py
# FastAPI backend for the Legal AI system
# Exposes endpoints for analysis, chat, contract storage and comparison

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from database import init_db, save_contract, get_all_contracts, get_contract_by_id, get_contracts_by_type
from agents.crew_agent import analyse_with_crew
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

init_db()  # Initialise database on startup

# Required for React frontend to communicate with this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Data shapes ---

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    contract_text: str
    question: str
    chat_history: list[Message] = []

class AnalysisResponse(BaseModel):
    contract_type: str
    answer: str
    tools_used: list[str]

class ChatResponse(BaseModel):
    answer: str
    tools_used: list[str]

class ContractRecord(BaseModel):
    id: int
    filename: str
    contract_type: str
    uploaded_at: str

class CompareRequest(BaseModel):
    contract_text: str
    contract_type: str


# --- Endpoints ---

@app.get("/health")
def health_check():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_contract(file: UploadFile = File(...)):
    """
    Accepts a PDF or DOCX contract file.
    Extracts the text, passes it to the legal agent, and returns a full analysis.
    Also saves the contract to the database for future comparison.
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

    contract_type = _detect_type(text)

    # Save to database for future comparison
    save_contract(
        filename=file.filename,
        contract_type=contract_type,
        text=text,
        summary=result["answer"]
    )

    return AnalysisResponse(
        contract_type=contract_type,
        answer=result["answer"],
        tools_used=result["tools_used"]
    )


@app.post("/chat", response_model=ChatResponse)
async def chat_with_contract(request: ChatRequest):
    """
    Accepts a contract, a question, and the full conversation history.
    The agent uses the history to answer follow-up questions in context.
    """
    if len(request.contract_text) < 50:
        raise HTTPException(status_code=422, detail="Contract text too short")

    # Convert Pydantic Message objects to plain dicts for the agent
    history = [{"role": msg.role, "content": msg.content} for msg in request.chat_history]

    try:
        result = run_agent(request.question, request.contract_text, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat failed: {e}")

    return ChatResponse(
        answer=result["answer"],
        tools_used=result["tools_used"]
    )


@app.get("/contracts", response_model=list[ContractRecord])
def list_contracts():
    """Returns all stored contracts."""
    return get_all_contracts()


@app.get("/contracts/{contract_id}")
def get_contract(contract_id: int):
    """Returns a single contract by id."""
    contract = get_contract_by_id(contract_id)
    if not contract:
        raise HTTPException(status_code=404, detail="Contract not found")
    return contract


@app.post("/compare")
async def compare_contract(request: CompareRequest):
    """
    Compares a new contract against all stored contracts of the same type.
    This is the knowledge graph feature — finds patterns across contract history.
    """
    previous = get_contracts_by_type(request.contract_type)

    if not previous:
        return {
            "answer": "No previous contracts of this type found. Upload more contracts to enable comparison.",
            "contracts_compared": 0
        }

    # Build comparison input — new contract + last 3 previous contracts
    comparison_input = f"NEW CONTRACT:\n{request.contract_text}\n\n"
    for i, contract in enumerate(previous[:3]):
        comparison_input += f"PREVIOUS CONTRACT {i+1} ({contract['filename']}):\n{contract['text']}\n\n"

    try:
        result = run_agent(
            "Compare the new contract against the previous contracts.",
            comparison_input
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison failed: {e}")

    return {
        "answer": result["answer"],
        "contracts_compared": len(previous[:3]),
        "tools_used": result["tools_used"]
    }

@app.post("/crew-analyse")
async def crew_analyse_contract(file: UploadFile = File(...)):
    """
    Full multi-agent analysis using CrewAI.
    4 specialist agents work in sequence:
    Contract Reader → Risk Analyst → UK Legal Expert → Senior Partner
    """
    if not file.filename.lower().endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX supported")

    suffix = "pdf" if file.filename.lower().endswith('.pdf') else "docx"
    temp_path = None

    try:
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

    try:
        # Run CrewAI in a thread pool to avoid blocking the async event loop
        import asyncio
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(None, analyse_with_crew, text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Crew analysis failed: {e}")

    return {
        "final_report": result["final_report"],
        "agents_used": result["agents_used"]
    }

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