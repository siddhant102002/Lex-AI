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
def compare_contracts(input_text: str) -> str:
    """
    Compares a new contract against one or more previous contracts.
    Use this when the user wants to know how this contract differs from others,
    or whether this contract is better or worse than previous ones.
    Input should contain the new contract followed by previous contracts clearly labelled.
    """
    return _call_claude(f"""You are a legal expert comparing contracts.

Analyse the contracts provided and give:
1. Key differences between them
2. Which differences favour which party
3. Any clauses that are unusually better or worse than the others
4. An overall verdict — is the new contract better or worse?

Be specific. Reference actual clause content.

{input_text}""")

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

@tool
def analyse_uk_jurisdiction(contract_text: str) -> str:
    """
    Analyses a contract specifically for UK law compliance and jurisdiction issues.
    Use this when the contract mentions UK, England, Wales, Scotland, British law,
    or when the user asks about UK-specific legal requirements, GDPR, or English law.
    """
    return _call_claude(f"""You are a legal assistant specialising in UK law. Please analyse this contract under UK legal standards.
                        
    Check for:

1. **GDPR / UK Data Protection Act 2018**
   - Does the contract handle personal data?
   - Are there adequate data protection clauses?
   - Is there a lawful basis for processing data?

2. **Employment Rights Act 1996** (if employment contract)
   - Is the notice period at least statutory minimum (1 week per year of service)?
   - Are redundancy rights mentioned?
   - Is unfair dismissal protection referenced?

3. **Working Time Regulations 1998** (if employment contract)
   - Is holiday entitlement at least 28 days?
   - Is the 48-hour working week limit addressed?

4. **Equality Act 2010**
   - Are there any potentially discriminatory clauses?

5. **Unfair Contract Terms Act 1977**
   - Are there any exclusion clauses that may be unenforceable?

6. **Governing Law**
   - Is English/Welsh/Scottish law specified?
   - Which courts have jurisdiction?
                        
7. **Renters Rights Act 2025** (if tenancy/rental agreement)
   - Are no-fault eviction clauses present? These are now illegal.
   - Is the tenancy periodic or fixed term? Fixed terms are now abolished.
   - Is rent increase procedure compliant with the new rules?
   - Is the property registered on the Private Rented Sector Database?

For each issue found:
- What the contract says
- What UK law requires
- Whether the contract complies
- What should be added or changed

Contract:
{contract_text}""")


# All 6 tools in one list — the agent loads this
ALL_TOOLS = [compare_contracts, summarize_contract, identify_clauses, flag_risks, suggest_questions, analyse_uk_jurisdiction]