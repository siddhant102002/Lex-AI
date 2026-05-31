import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from tools.contract_tools import ALL_TOOLS

load_dotenv()


def create_legal_agent():
    # Claude as the brain, with tools bound directly
    llm = ChatAnthropic(
        model="claude-sonnet-4-5",
        api_key=os.getenv("ANTHROPIC_API_KEY"),
        max_tokens=4000
    )
    llm_with_tools = llm.bind_tools(ALL_TOOLS)
    return llm_with_tools


llm_with_tools = create_legal_agent()


def run_agent(user_question: str, contract_text: str, chat_history: list = []) -> dict:

    messages = [
        SystemMessage(content="""You are an expert legal AI assistant.
You help users understand legal contracts.
Think carefully about what the user needs.
Always write in plain English.
You have memory of the entire conversation — use it to answer follow-up questions.""")
    ]

    messages.append(HumanMessage(content=f"""Here is the contract I want to discuss:
---START---
{contract_text}
---END---"""))

    for msg in chat_history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(HumanMessage(content=user_question))

    # First call — agent decides which tools to use
    response = llm_with_tools.invoke(messages)

    tools_used = []
    tool_results = []

    if hasattr(response, "tool_calls") and response.tool_calls:
        for tool_call in response.tool_calls:
            tools_used.append(tool_call["name"])
            print(f"Running tool: {tool_call['name']}")

            # Find and run the matching tool
        for tool in ALL_TOOLS:
                if tool.name == tool_call["name"]:
                    try:
                        args = tool_call["args"]
                        if tool.name == "compare_contracts":
                            if "input_text" not in args:
                                args["input_text"] = contract_text
                        else:
                            if "contract_text" not in args:
                                args["contract_text"] = contract_text
                        result = tool.invoke(args)
                        print(f"Tool result preview: {str(result)[:200]}")
                        tool_results.append(f"### {tool_call['name']}\n{result}")
                    except Exception as e:
                        print(f"Tool {tool_call['name']} failed: {e}")
                        tool_results.append(f"### {tool_call['name']}\nError: {e}")

        # Combine all tool results into the final answer
        answer = "\n\n".join(tool_results)

    else:
        # No tools used — answer came directly from Claude
        if isinstance(response.content, list):
            answer = " ".join([block.text for block in response.content if hasattr(block, "text")])
        else:
            answer = response.content

    return {
        "answer": answer,
        "tools_used": tools_used
    }