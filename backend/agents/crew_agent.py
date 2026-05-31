# crew_agent.py
# Multi-agent system using CrewAI
# Four specialist agents work in sequence — each builds on the previous agent's work

from crewai import Agent, Task, Crew, Process, LLM
import os
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env'))

api_key = os.getenv("ANTHROPIC_API_KEY")
if not api_key:
    raise ValueError("ANTHROPIC_API_KEY not found. Check your .env file in the backend folder.")

llm = LLM(
    model="anthropic/claude-sonnet-4-5",
    api_key=api_key
)

# AGENT 1 — reads and understands the contract
contract_reader = Agent(
    role="Contract Reader",
    goal="Extract and clearly understand all key information from the contract",
    backstory="""You are a paralegal with 10 years of experience reading legal contracts.
You are thorough, precise, and extract every important detail before passing
the work to specialist colleagues.""",
    llm=llm,
    verbose=True
)

# AGENT 2 — finds all risks and unfair clauses
risk_analyst = Agent(
    role="Risk Analyst",
    goal="Identify all risks, unfair clauses, and red flags in the contract",
    backstory="""You are a legal risk specialist who has reviewed thousands of contracts.
You have a sharp eye for one-sided clauses, hidden liabilities, and unfair terms.
You always explain why something is risky and what a fairer version looks like.""",
    llm=llm,
    verbose=True
)

# AGENT 3 — checks UK law compliance
uk_legal_expert = Agent(
    role="UK Legal Expert",
    goal="Check the contract for compliance with UK law and flag any violations",
    backstory="""You are a UK solicitor specialising in contract law.
You know the Employment Rights Act 1996, Working Time Regulations 1998,
UK GDPR, Equality Act 2010, and Renters Rights Act 2025 inside out.
You immediately spot clauses that would not hold up in English courts.""",
    llm=llm,
    verbose=True
)

# AGENT 4 — reviews everything and produces the final report
senior_partner = Agent(
    role="Senior Partner",
    goal="Review all specialist analyses and produce a clear final recommendation",
    backstory="""You are a senior partner at a top UK law firm with 20 years experience.
You take input from your team of specialists and synthesise it into a clear,
actionable report that a non-lawyer can understand and act on.""",
    llm=llm,
    verbose=True
)


def analyse_with_crew(contract_text: str) -> dict:
    """
    Runs a full multi-agent analysis on a contract.
    Four specialists work in sequence — each building on the previous agent's work.
    """

    task_read = Task(
        description=f"""Read this contract carefully and extract:
1. Who the parties are
2. What the contract is about
3. Key obligations of each party
4. Duration and termination terms
5. Any unusual or notable provisions

Contract:
{contract_text}""",
        agent=contract_reader,
        expected_output="A clear structured summary of the contract's key information"
    )

    task_risks = Task(
        description="""Using the contract summary from the Contract Reader,
identify all risks, red flags, and unfair clauses.
For each risk explain what it says, why it is risky, and what a fairer version looks like.""",
        agent=risk_analyst,
        expected_output="A detailed list of risks with explanations and suggested improvements"
    )

    task_uk = Task(
        description="""Using the contract summary, check for UK law compliance.
Check against: Employment Rights Act 1996, Working Time Regulations 1998,
UK GDPR, Equality Act 2010, Unfair Contract Terms Act 1977, Renters Rights Act 2025.
Flag any violations and explain what UK law requires.""",
        agent=uk_legal_expert,
        expected_output="A UK law compliance report with specific violations and recommendations"
    )

    task_final = Task(
        description="""Review the analyses from the Contract Reader, Risk Analyst,
and UK Legal Expert. Produce a final report with:
1. Executive summary (3 sentences max)
2. Top 5 most important issues
3. Overall risk rating (Low / Medium / High / Critical)
4. Clear recommendation — sign, negotiate, or reject
5. Priority actions before signing""",
        agent=senior_partner,
        expected_output="A clear final report with recommendation and priority actions"
    )

    crew = Crew(
        agents=[contract_reader, risk_analyst, uk_legal_expert, senior_partner],
        tasks=[task_read, task_risks, task_uk, task_final],
        process=Process.sequential,
        verbose=True
    )

    try:
        result = crew.kickoff()
    except Exception as e:
        print(f"Crew failed with error: {e}")
        raise e

    return {
        "final_report": str(result),
        "agents_used": ["Contract Reader", "Risk Analyst", "UK Legal Expert", "Senior Partner"]
    }