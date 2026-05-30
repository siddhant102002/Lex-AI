# contract_tools.py
# This file contains the 4 skills (tools) the agent can use
# Each function is wrapped with @tool so the agent can find and use them

import anthropic
import os
from dotenv import load_dotenv
from langchain.tools import tool  

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-5"


def _call_claude(prompt: str) -> str:
    """Calls Claude and returns the text response."""
    messages = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return messages.content[0].text


@tool
def summarize_contract(contract_text: str) -> str:
    """
    Summarises a legal contract in plain English.
    Use this when the user wants an overview or summary of the contract.
    """
    return _call_claude(f"""You are a legal assistant. Please summarize this contract in plain English.
                
Cover:
1. What the contract is about
2. Who the parties are
3. What each party must do
4. How long it lasts
5. How it ends
            
Contract:
{contract_text}""")


@tool
def identify_clauses(contract_text: str) -> str:
    """
    Identifies and explains all key clauses in a legal contract.
    Use this when the user wants to understand specific clauses or provisions.
    """
    return _call_claude(f"""You are a legal assistant. Please identify the key clauses in this contract.

For each clause:
1. Clause Name
2. What It Says (plain English)
3. Why It Matters

Contract:
{contract_text}""")


@tool
def flag_risks(contract_text: str) -> str:
    """
    Identifies risky or one-sided clauses in a legal contract.
    Use this when the user asks about risks, red flags, or what to watch out for before signing.
    """
    return _call_claude(f"""You are a legal assistant. Please review this contract and flag any risks.

For each risk:
1. What it says
2. Why it is risky
3. What a fairer version would look like

Contract:
{contract_text}""")


@tool
def suggest_questions(contract_text: str) -> str:
    """
    Generates important questions to ask before signing a contract.
    Use this when the user wants to know what to ask or how to negotiate.
    """
    return _call_claude(f"""You are a legal assistant. Please list 10 questions 
someone should ask before signing this contract.

Contract:
{contract_text}""")


# All 4 tools in one list — the agent loads this
ALL_TOOLS = [summarize_contract, identify_clauses, flag_risks, suggest_questions]