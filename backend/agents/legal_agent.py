import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from tools.contract_tools import ALL_TOOLS

load_dotenv()


def create_legal_agent():
    # Claude as the brain, with tools bound directly
    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4000
    )

    # Bind tools directly to the model — new LangChain approach
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    return llm_with_tools


llm_with_tools = create_legal_agent()


def run_agent(user_question: str, contract_text: str) -> dict:

    full_input = f"""Here is the contract:
---START---
{contract_text}
---END---

Question: {user_question}"""

    messages = [
        SystemMessage(content="""You are an expert legal AI assistant.
You help users understand legal contracts.
Think carefully about what the user needs.
Choose the RIGHT tool for their specific question.
Always write in plain English."""),
        HumanMessage(content=full_input)
    ]

    response = llm_with_tools.invoke(messages)

    # Extract tool calls if any were made
    tools_used = []
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tools_used.append(tool_call["name"])

    return {
        "answer": response.content,
        "tools_used": tools_used
    }