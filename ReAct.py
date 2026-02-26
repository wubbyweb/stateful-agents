"""
ReAct Agent — Built from scratch to understand the fundamentals.
This agent can reason about tasks and use tools to accomplish them.
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


# ── System prompt that teaches the LLM the ReAct pattern ──────────
REACT_SYSTEM_PROMPT = """You are an AI assistant that uses the ReAct
(Reasoning and Acting) framework. For each user request, you will:

1. THOUGHT: Reason about what you need to do next
2. ACTION: Choose a tool to use (or "finish" if done)
3. OBSERVATION: Review the tool's output

Available tools:
- search_database(query: str) → Search records
- calculate(expression: str) → Evaluate math
- finish(answer: str) → Return final answer

Respond in this EXACT format:
THOUGHT: <your reasoning>
ACTION: <tool_name>(<arguments>)

When you have enough information:
THOUGHT: <your final reasoning>
ACTION: finish(<your complete answer>)
"""

def search_database(query: str) -> str:
    """Simulated database search."""
    data = {
        "q3 sales": "$2.4M total — North: $1.1M, South: $800K, West: $500K",
        "top customers": "1. Acme Corp ($400K), 2. Globex ($350K), 3. Initech ($200K)",
        "inventory": "Widget A: 1,200 units, Widget B: 850 units, Widget C: 2,100 units",
    }
    for key, value in data.items():
        if key in query.lower():
            return value
    return "No results found."

def calculate(expression: str) -> str:
    """Safe math evaluator."""
    try:
        result = eval(expression, {"__builtins__": {}})  # restricted eval
        return str(result)
    except Exception as e:
        return f"Error: {e}"

TOOLS = {
    "search_database": search_database,
    "calculate": calculate,
}

def run_react_agent(user_query: str, max_steps: int = 10) -> str:
    """Execute the ReAct loop for a given user query."""
    messages = [
        {"role": "system", "content": REACT_SYSTEM_PROMPT},
        {"role": "user",   "content": user_query},
    ]

    for step in range(max_steps):
        # ── THINK + ACT: Ask the LLM what to do ──
        response = client.chat.completions.create(
            model="llama3:8b",
            messages=messages,
            temperature=0,
        )
        assistant_msg = response.choices[0].message.content
        messages.append({"role": "assistant", "content": assistant_msg})

        print(f"\n── Step {step + 1} ──")
        print(assistant_msg)

        # ── Parse the action ──
        if "ACTION: finish(" in assistant_msg:
            # Extract final answer
            answer = assistant_msg.split("ACTION: finish(")[1].rstrip(")")
            return answer

        if "ACTION:" in assistant_msg:
            action_line = assistant_msg.split("ACTION:")[1].strip()
            tool_name = action_line.split("(")[0].strip()
            args_str  = action_line.split("(", 1)[1].rstrip(")")

            # ── OBSERVE: Execute the tool ──
            if tool_name in TOOLS:
                observation = TOOLS[tool_name](args_str.strip('"').strip("'"))
            else:
                observation = f"Error: Unknown tool '{tool_name}'"

            print(f"OBSERVATION: {observation}")
            messages.append({
                "role": "user",
                "content": f"OBSERVATION: {observation}"
            })

    return "Max steps reached without finding an answer."

# ── Run it ──
if __name__ == "__main__":
    answer = run_react_agent("What were our Q3 sales and who were the top customers?")
    print(f"\n{'='*50}")
    print(f"FINAL ANSWER: {answer}")