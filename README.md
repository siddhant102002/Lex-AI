# ⚖️ Lex AI — Agentic Legal Contract Intelligence

> Multi-agent Legal AI system that analyses contracts, flags risks, checks UK law compliance, and delivers a comprehensive 4-agent deep review.

---

## What it does

Most contract analysis tools just summarise. Lex AI thinks through contracts the way a law firm does.

Upload any contract and get:

- **Plain English Summary** — who the parties are, what each must do, how it ends
- **Key Clauses** — every clause explained with why it matters
- **Risk Flags** — one-sided terms, liability gaps, unfair clauses with fairer alternatives
- **UK Law Compliance** — checks against GDPR, Employment Rights Act 1996, Working Time Regulations 1998, Equality Act 2010, Renters Rights Act 2025
- **4-Agent Deep Review** — Contract Reader → Risk Analyst → UK Legal Expert → Senior Partner
- **Chat with Contract** — ask questions with full conversation memory
- **Contract Comparison** — compares against your history to find risks across multiple documents

---

## Architecture

Frontend (Vanilla JS)
↓
FastAPI Backend
↓
LangChain Agent          CrewAI Multi-Agent Pipeline
6 specialist tools       Contract Reader → Risk Analyst
→ UK Legal Expert → Senior Partner
↓
Claude AI    SQLite (contract history)
↓
Docker → HuggingFace Spaces

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Python, FastAPI |
| AI Agents | LangChain, CrewAI |
| LLM | Anthropic Claude |
| Database | SQLite |
| Frontend | Vanilla JS, HTML, CSS |
| Deployment | Docker, HuggingFace Spaces |

---

## UK Law Coverage

- UK GDPR / Data Protection Act 2018
- Employment Rights Act 1996
- Working Time Regulations 1998
- Equality Act 2010
- Unfair Contract Terms Act 1977
- Renters Rights Act 2025

---

## Run Locally

```bash
git clone https://github.com/siddhant102002/legal-ai-v2
cd legal-ai-v2
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

Open `http://localhost:8000`

---

## Built by

**Siddhant Tayade** — MSc AI, University of York

[LinkedIn](https://linkedin.com/in/siddhant-tayade-1b92a2396) · [GitHub](https://github.com/siddhant102002) 
