# contract_tools.py
import anthropic
import os
from dotenv import load_dotenv
from langchain.tools import tool  

load_dotenv()

client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
MODEL = "claude-sonnet-4-5"


def _call_claude(prompt: str) -> str:
    messages = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}]
    )
    return messages.content[0].text


@tool
def summarize_contract(contract_text: str) -> str:
    """Summarises a legal contract in plain English."""
    return _call_claude(f"""You are a legal expert. Summarise this contract in plain English.

Use this exact markdown structure:
# [Contract Type] Summary

## 1. What the contract is about
[explanation]

## 2. Who the parties are
- **Party name**: description

## 3. What each party must do

**[Party 1] must:**
- obligation 1
- obligation 2

**[Party 2] must:**
- obligation 1

## 4. How long it lasts
[explanation]

## 5. How it ends
[explanation]

Contract:
{contract_text}""")


@tool
def identify_clauses(contract_text: str) -> str:
    """Identifies and explains all key clauses in a legal contract."""
    return _call_claude(f"""You are a legal expert. Identify all key clauses in this contract.

Use this exact markdown structure for each clause:

## [Clause Name]

### What It Says
[plain English explanation]

### Why It Matters
[why this clause is important]

---

Repeat for every key clause. Look for: payment terms, termination, liability, confidentiality, intellectual property, dispute resolution, governing law, data protection.

Contract:
{contract_text}""")


@tool
def flag_risks(contract_text: str) -> str:
    """Identifies risky or one-sided clauses in a legal contract."""
    return _call_claude(f"""You are a legal expert reviewing this contract for risks.

Use this exact markdown structure:

# Contract Risk Analysis

## ⚠️ HIGH RISKS

### Risk 1: [Risk Name]

#### The Problem
[what the clause says]

#### Why It's Risky
- reason 1
- reason 2

#### Fairer Version
[what a balanced clause would say]

---

Repeat for every risk found. Organise by: HIGH RISKS, MEDIUM RISKS, LOWER RISKS.

End with:
## Summary
- Overall risk level: HIGH / MEDIUM / LOW
- Top 3 concerns
- Recommendation: sign / negotiate / reject

Contract:
{contract_text}""")


@tool
def suggest_questions(contract_text: str) -> str:
    """Generates important questions to ask before signing a contract."""
    return _call_claude(f"""You are a legal expert. List 10 critical questions to ask before signing.

Use this structure:
# 10 Questions to Ask Before Signing

## 1. [Question title]
[Why this question matters and what to look for in the answer]

Repeat for all 10 questions.

Contract:
{contract_text}""")


@tool
def compare_contracts(input_text: str) -> str:
    """Compares a new contract against previous contracts."""
    return _call_claude(f"""You are a legal expert comparing contracts.

# Contract Comparison

## Key Differences
[list main differences]

## Which Party Benefits
[explain who gains from each difference]

## Overall Verdict
[is the new contract better or worse]

{input_text}""")


@tool
def analyse_uk_jurisdiction(contract_text: str) -> str:
    """Analyses a contract for UK law compliance."""
    return _call_claude(f"""You are a UK solicitor. Analyse this contract for UK law compliance.

# UK Law Compliance Report

## UK GDPR / Data Protection Act 2018
### Status: [COMPLIANT / NON-COMPLIANT / NOT APPLICABLE]
[findings]

## Employment Rights Act 1996
### Status: [COMPLIANT / NON-COMPLIANT / NOT APPLICABLE]
[findings]

## Working Time Regulations 1998
### Status: [COMPLIANT / NON-COMPLIANT / NOT APPLICABLE]
[findings]

## Equality Act 2010
### Status: [COMPLIANT / NON-COMPLIANT / NOT APPLICABLE]
[findings]

## Unfair Contract Terms Act 1977
### Status: [COMPLIANT / NON-COMPLIANT / NOT APPLICABLE]
[findings]

## Renters Rights Act 2025
### Status: [COMPLIANT / NON-COMPLIANT / NOT APPLICABLE]
[findings]

## Overall Compliance Summary
- Compliant areas: [list]
- Violations found: [list]
- Recommended actions: [list]

Contract:
{contract_text}""")


ALL_TOOLS = [compare_contracts, summarize_contract, identify_clauses, flag_risks, suggest_questions, analyse_uk_jurisdiction]