# 🧠 Mastering Stateful Multi-Agent AI Systems

> **A comprehensive learning guide** for designing stateful systems where multiple AI agents collaborate to execute complex, multi-step workflows.

---

## 📋 Table of Contents

1. [Learning Roadmap](#-learning-roadmap)
2. [Core Concepts](#-core-concepts)
3. [The ReAct Loop — Reasoning & Acting](#-the-react-loop--reasoning--acting)
4. [Framework Deep Dives](#-framework-deep-dives)
   - [LangGraph](#1-langgraph)
   - [Semantic Kernel](#2-semantic-kernel)
   - [CrewAI](#3-crewai)
   - [Microsoft AutoGen](#4-microsoft-autogen)
5. [Multi-Agent Collaboration Patterns](#-multi-agent-collaboration-patterns)
6. [Database Querying & Dynamic Memory](#-database-querying--dynamic-memory)
7. [Production Best Practices](#-production-best-practices)
8. [Hands-On Learning Exercises](#-hands-on-learning-exercises)
9. [Glossary](#-glossary)

---

## 🗺 Learning Roadmap

```
Level 1: Foundations              Level 2: Frameworks           Level 3: Mastery
┌─────────────────────┐      ┌──────────────────────┐      ┌──────────────────────┐
│ • What is "state"?  │      │ • LangGraph          │      │ • Production deploy  │
│ • Agent anatomy     │──▶   │ • Semantic Kernel     │──▶   │ • Observability      │
│ • ReAct loop        │      │ • CrewAI              │      │ • Error recovery     │
│ • Tool use          │      │ • AutoGen             │      │ • Scaling patterns   │
│ • Memory types      │      │ • Choosing a framework│      │ • Security & eval    │
└─────────────────────┘      └──────────────────────┘      └──────────────────────┘
```

| Phase | Duration | Focus |
|-------|----------|-------|
| **Foundations** | 1–2 weeks | Understand state, agents, ReAct, tool-use |
| **Single Agent** | 1–2 weeks | Build agents with one framework |
| **Multi-Agent** | 2–3 weeks | Orchestrate collaborating agents |
| **Production** | 2–4 weeks | Deploy, monitor, and harden systems |

---

## 🧱 Core Concepts

### What Is a "Stateful" Agent?

A **stateless** system forgets everything between requests — like a vending machine that doesn't remember your last purchase. A **stateful** system maintains context across interactions — like a personal assistant who remembers your preferences, ongoing tasks, and past conversations.

```
┌──────────────────────────────────────────────────────┐
│                   STATEFUL AGENT                     │
│                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐ │
│  │  Inputs  │──▶│  Brain   │──▶│   Outputs        │ │
│  │ (prompts,│   │ (LLM +   │   │ (actions, text,  │ │
│  │  context)│   │  logic)  │   │  tool calls)     │ │
│  └──────────┘   └────┬─────┘   └──────────────────┘ │
│                      │                               │
│                 ┌────▼─────┐                         │
│                 │  STATE   │                         │
│                 │ (memory, │                         │
│                 │  history,│                         │
│                 │  plans)  │                         │
│                 └──────────┘                         │
└──────────────────────────────────────────────────────┘
```

### The Three Pillars of Stateful Agent Systems

#### 1. State Management

State is the data that persists across an agent's reasoning steps. It includes:

| State Type | Description | Example |
|------------|-------------|---------|
| **Conversation History** | Past messages between user and agent | Chat logs |
| **Working Memory** | Temporary data for current task | Intermediate calculations |
| **Long-Term Memory** | Persistent knowledge across sessions | User preferences, learned facts |
| **Task State** | Progress through a multi-step plan | "Step 3 of 7 complete" |
| **Shared State** | Data accessible to multiple agents | A shared scratchpad or database |

```python
# Example: A simple state object
from dataclasses import dataclass, field
from typing import Any

@dataclass
class AgentState:
    """The state that persists across agent reasoning steps."""
    messages: list[dict] = field(default_factory=list)
    working_memory: dict[str, Any] = field(default_factory=dict)
    task_plan: list[str] = field(default_factory=list)
    current_step: int = 0
    tools_used: list[str] = field(default_factory=list)

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def update_memory(self, key: str, value: Any):
        self.working_memory[key] = value

    def advance_step(self):
        self.current_step += 1
```

#### 2. Tool Use

Agents become powerful when they can **act** on the world — querying databases, calling APIs, running code, reading files. Tools are functions the LLM can invoke.

```python
# Example: Defining tools for an agent
import json

def search_database(query: str) -> str:
    """Search the customer database for matching records."""
    # Simulated database query
    results = [
        {"id": 1, "name": "Alice", "status": "active"},
        {"id": 2, "name": "Bob",   "status": "inactive"},
    ]
    matches = [r for r in results if query.lower() in r["name"].lower()]
    return json.dumps(matches)

def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient."""
    # In production, this would call an email API
    return f"Email sent to {to} with subject: {subject}"

def update_record(record_id: int, field: str, value: str) -> str:
    """Update a specific field in a database record."""
    return f"Record {record_id}: set {field} = {value}"

# Tool registry — agents pick from this menu
TOOLS = {
    "search_database": search_database,
    "send_email": send_email,
    "update_record": update_record,
}
```

#### 3. Orchestration

Orchestration is the **conductor** that decides which agent acts, when, and with what information. It manages:

- **Routing**: Which agent handles which subtask
- **Sequencing**: The order of operations
- **Conflict resolution**: What happens when agents disagree
- **Error handling**: Recovery from failures

```
          ┌──────────────┐
          │ ORCHESTRATOR │
          │              │
          │  • Routes    │
          │  • Sequences │
          │  • Monitors  │
          └──────┬───────┘
                 │
       ┌─────────┼─────────┐
       ▼         ▼         ▼
  ┌─────────┐ ┌─────────┐ ┌─────────┐
  │ Agent A │ │ Agent B │ │ Agent C │
  │Research │ │Analysis │ │ Writing │
  └────┬────┘ └────┬────┘ └────┬────┘
       │           │           │
       └─────────┐ │ ┌─────────┘
                 ▼ ▼ ▼
           ┌──────────────┐
           │ SHARED STATE │
           └──────────────┘
```

---

## 🔄 The ReAct Loop — Reasoning & Acting

### What Is ReAct?

**ReAct** (Reasoning and Acting) is a paradigm where an LLM alternates between **thinking** (reasoning about what to do) and **acting** (executing tools or actions). This was formalized in the paper *"ReAct: Synergizing Reasoning and Acting in Language Models"* (Yao et al., 2022).

### The Core Loop

```
┌──────────────────────────────────────────────────┐
│                  ReAct Loop                      │
│                                                  │
│   ┌──────────┐                                   │
│   │ OBSERVE  │◀──────────────────────┐           │
│   │ (read    │                       │           │
│   │  state)  │                       │           │
│   └────┬─────┘                       │           │
│        ▼                             │           │
│   ┌──────────┐                       │           │
│   │  THINK   │     ┌──────────┐      │           │
│   │ (reason  │────▶│   ACT    │──────┘           │
│   │  & plan) │     │ (execute │                  │
│   └──────────┘     │  tool)   │                  │
│        │           └──────────┘                  │
│        ▼ (if done)                               │
│   ┌──────────┐                                   │
│   │  ANSWER  │                                   │
│   └──────────┘                                   │
└──────────────────────────────────────────────────┘
```

### Step-by-Step Breakdown

| Step | What Happens | Example |
|------|-------------|---------|
| **1. Observe** | Agent reads current state, user input, tool results | "User asked: What were Q3 sales?" |
| **2. Think** | Agent reasons about what to do next | "I need to query the sales database for Q3 data" |
| **3. Act** | Agent calls a tool or takes an action | `search_database("Q3 sales totals")` |
| **4. Observe** | Agent reads the tool's result | "Got result: $2.4M total revenue" |
| **5. Think** | Agent reasons whether it has enough info | "I have the total. Let me break it down by region." |
| **6. Act** | Agent calls another tool | `search_database("Q3 sales by region")` |
| **7. Answer** | Agent formulates the final response | "Q3 sales were $2.4M. Here's the breakdown..." |

### Implementation from Scratch

```python
"""
ReAct Agent — Built from scratch to understand the fundamentals.
This agent can reason about tasks and use tools to accomplish them.
"""
from openai import OpenAI

client = OpenAI()

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
            model="gpt-4o",
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
```

### Key Insight: Why ReAct Matters

> **Without ReAct**, an LLM tries to answer in a single shot — often hallucinating facts it doesn't know.
>
> **With ReAct**, the LLM acknowledges gaps in its knowledge, uses tools to gather real data, and builds an answer incrementally from verified information.

This is the **foundation** of every framework covered below. LangGraph, CrewAI, Semantic Kernel, and AutoGen all implement variations of the ReAct loop.

---

## 🛠 Framework Deep Dives

### 1. LangGraph

**What**: A library from LangChain for building stateful, multi-actor applications as **graphs**. Each node is a function, and edges define the flow (including conditional routing).

**Best For**: Complex workflows with conditional branching, cycles, human-in-the-loop, and fine-grained state control.

**Install**:
```bash
pip install langgraph langchain-openai
```

#### Mental Model: Graphs, Not Chains

```
Traditional Chain:             LangGraph:
A ──▶ B ──▶ C ──▶ D           A ──▶ B ──┬──▶ C
                                         │
                                         └──▶ D ──▶ B  (cycle!)
                                              ▲
                                              │ (conditional)
```

LangGraph models workflows as **directed graphs** with:
- **Nodes**: Functions that transform state
- **Edges**: Connections between nodes (can be conditional)
- **State**: A typed object that flows through the graph

#### Example: Research Agent with LangGraph

```python
"""
A research agent that plans, searches, analyzes, and writes reports.
Demonstrates: state management, conditional routing, cycles.
"""
from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

# ── Step 1: Define the State ──────────────────────────────────────
class ResearchState(TypedDict):
    """All data flowing through the graph."""
    messages: Annotated[list, add_messages]  # Chat history (auto-appended)
    research_topic: str                      # What we're researching
    search_results: list[str]                # Raw search data
    analysis: str                            # Processed findings
    report: str                              # Final output
    iteration: int                           # Loop counter

# ── Step 2: Define Node Functions ─────────────────────────────────
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def plan_research(state: ResearchState) -> dict:
    """The agent decides what to search for."""
    response = llm.invoke([
        SystemMessage(content="You are a research planner. Given a topic, "
                      "output 3 specific search queries, one per line."),
        HumanMessage(content=f"Topic: {state['research_topic']}")
    ])
    return {
        "messages": [response],
        "search_results": [],
        "iteration": state.get("iteration", 0),
    }

def execute_search(state: ResearchState) -> dict:
    """Simulated web search (replace with real API in production)."""
    queries = state["messages"][-1].content.strip().split("\n")
    results = [f"Result for '{q}': [simulated data about {q}]" for q in queries]
    return {"search_results": results}

def analyze_results(state: ResearchState) -> dict:
    """The agent analyzes search results and decides if more research is needed."""
    results_text = "\n".join(state["search_results"])
    response = llm.invoke([
        SystemMessage(content="Analyze these search results. Summarize key findings. "
                      "End with either SUFFICIENT or NEED_MORE_DATA."),
        HumanMessage(content=results_text)
    ])
    return {
        "messages": [response],
        "analysis": response.content,
        "iteration": state["iteration"] + 1,
    }

def write_report(state: ResearchState) -> dict:
    """Generate the final report."""
    response = llm.invoke([
        SystemMessage(content="Write a concise research report based on this analysis."),
        HumanMessage(content=state["analysis"])
    ])
    return {"report": response.content, "messages": [response]}

# ── Step 3: Define Routing Logic ──────────────────────────────────
def should_continue_research(state: ResearchState) -> Literal["search_more", "write"]:
    """Conditional edge: decide whether to search more or write the report."""
    if state["iteration"] >= 3:
        return "write"  # Safety limit
    if "NEED_MORE_DATA" in state.get("analysis", ""):
        return "search_more"
    return "write"

# ── Step 4: Build the Graph ───────────────────────────────────────
workflow = StateGraph(ResearchState)

# Add nodes
workflow.add_node("plan",    plan_research)
workflow.add_node("search",  execute_search)
workflow.add_node("analyze", analyze_results)
workflow.add_node("write",   write_report)

# Add edges
workflow.set_entry_point("plan")
workflow.add_edge("plan",   "search")
workflow.add_edge("search", "analyze")

# Conditional edge — this is where the magic happens
workflow.add_conditional_edges(
    "analyze",
    should_continue_research,
    {"search_more": "plan", "write": "write"},  # Cycle back or proceed
)
workflow.add_edge("write", END)

# Compile
app = workflow.compile()

# ── Step 5: Run ───────────────────────────────────────────────────
result = app.invoke({
    "research_topic": "Impact of AI agents on software development productivity",
    "messages": [],
    "search_results": [],
    "analysis": "",
    "report": "",
    "iteration": 0,
})
print(result["report"])
```

#### LangGraph Key Concepts Summary

| Concept | Description |
|---------|-------------|
| `StateGraph` | The graph builder — you add nodes and edges to this |
| `TypedDict` state | A typed dictionary that flows through every node |
| `add_messages` | A reducer that appends messages instead of replacing |
| Conditional edges | Routes based on state (enables loops and branching) |
| `compile()` | Freezes the graph into a runnable application |
| Checkpointing | Persist state to resume later (built-in SQLite/Postgres) |

#### LangGraph Checkpointing (Persistence)

```python
from langgraph.checkpoint.sqlite import SqliteSaver

# Add persistence — state survives restarts
memory = SqliteSaver.from_conn_string(":memory:")  # or a file path
app = workflow.compile(checkpointer=memory)

# Run with a thread_id to track state across invocations
config = {"configurable": {"thread_id": "research-session-1"}}
result = app.invoke(initial_state, config)

# Later, resume the same session:
result = app.invoke({"messages": [HumanMessage("Tell me more")]}, config)
```

---

### 2. Semantic Kernel

**What**: Microsoft's open-source SDK for integrating LLMs into applications. It emphasizes a **plugin architecture** where capabilities are modular, reusable "skills."

**Best For**: Enterprise applications, .NET ecosystems, structured plugin development, and integration with Azure AI services.

**Install**:
```bash
pip install semantic-kernel
```

#### Mental Model: Kernel + Plugins + Planner

```
┌──────────────────────────────────────────┐
│               KERNEL                     │
│                                          │
│  ┌──────────┐  ┌──────────┐  ┌────────┐ │
│  │ Plugin A │  │ Plugin B │  │Plugin C│ │
│  │(Database)│  │ (Email)  │  │(Math)  │ │
│  └──────────┘  └──────────┘  └────────┘ │
│                                          │
│  ┌──────────────────────────────────┐    │
│  │           PLANNER                │    │
│  │  Decomposes goals into steps     │    │
│  │  using available plugins         │    │
│  └──────────────────────────────────┘    │
│                                          │
│  ┌──────────────────────────────────┐    │
│  │           MEMORY                 │    │
│  │  Semantic + episodic memory      │    │
│  └──────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

#### Example: Multi-Plugin Agent

```python
"""
Semantic Kernel agent with plugins for database queries and email.
Demonstrates: Kernel setup, plugin creation, auto function calling.
"""
import asyncio
import semantic_kernel as sk
from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import kernel_function

# ── Step 1: Create Plugins ────────────────────────────────────────
class DatabasePlugin:
    """Plugin for database operations."""

    @kernel_function(
        name="query_customers",
        description="Search for customers by name or status"
    )
    def query_customers(self, search_term: str) -> str:
        """Query the customer database."""
        records = [
            {"name": "Alice Johnson", "status": "active",   "revenue": "$50K"},
            {"name": "Bob Smith",     "status": "inactive", "revenue": "$30K"},
            {"name": "Carol White",   "status": "active",   "revenue": "$75K"},
        ]
        matches = [r for r in records
                   if search_term.lower() in str(r).lower()]
        if matches:
            return str(matches)
        return "No matching customers found."

    @kernel_function(
        name="get_sales_summary",
        description="Get sales summary for a given quarter"
    )
    def get_sales_summary(self, quarter: str) -> str:
        """Retrieve sales data for the specified quarter."""
        data = {"Q1": "$1.2M", "Q2": "$1.8M", "Q3": "$2.4M", "Q4": "$3.1M"}
        return data.get(quarter.upper(), "Quarter not found.")


class EmailPlugin:
    """Plugin for email operations."""

    @kernel_function(
        name="send_report",
        description="Send a report via email to a recipient"
    )
    def send_report(self, recipient: str, subject: str, body: str) -> str:
        return f"✅ Report sent to {recipient}: '{subject}'"


# ── Step 2: Set Up the Kernel ─────────────────────────────────────
async def main():
    kernel = sk.Kernel()

    # Add the LLM service
    kernel.add_service(
        OpenAIChatCompletion(service_id="chat", ai_model_id="gpt-4o")
    )

    # Register plugins
    kernel.add_plugin(DatabasePlugin(), plugin_name="Database")
    kernel.add_plugin(EmailPlugin(),    plugin_name="Email")

    # ── Step 3: Enable Auto Function Calling ──────────────────────
    settings = kernel.get_prompt_execution_settings_from_service_id("chat")
    settings.function_choice_behavior = (
        sk.FunctionChoiceBehavior.Auto()  # LLM picks which plugins to call
    )

    # ── Step 4: Run a Conversation ────────────────────────────────
    chat_history = ChatHistory()
    chat_history.add_system_message(
        "You are a helpful business assistant. Use the Database and Email "
        "plugins to answer questions and take actions."
    )
    chat_history.add_user_message(
        "Find all active customers and email a summary report to manager@company.com"
    )

    result = await kernel.invoke_prompt(
        prompt="{{$chat_history}}",
        chat_history=chat_history,
        settings=settings,
    )
    print(result)

asyncio.run(main())
```

#### Semantic Kernel Key Concepts

| Concept | Description |
|---------|-------------|
| **Kernel** | The central runtime that connects LLMs, plugins, and memory |
| **Plugin** | A class of related functions the AI can call |
| `@kernel_function` | Decorator that exposes a Python function to the AI |
| **Planner** | Auto-decomposes complex goals into plugin call sequences |
| **Function Choice** | `Auto()` lets the LLM decide which functions to invoke |
| **Memory** | Semantic search over stored facts and conversation history |

---

### 3. CrewAI

**What**: A framework for orchestrating **role-playing AI agents** that work together as a "crew." Each agent has a specific role, goal, and backstory.

**Best For**: Task-oriented multi-agent systems, content creation pipelines, business process automation.

**Install**:
```bash
pip install crewai crewai-tools
```

#### Mental Model: Crew → Agents → Tasks

```
┌──────────────────────────────────────────────────┐
│                    CREW                          │
│                                                  │
│  ┌────────────────────────────────────────┐      │
│  │  Agent: Senior Researcher             │      │
│  │  Role: "Research analyst"             │      │
│  │  Goal: "Find accurate data"           │      │
│  │  Tools: [search, database_query]      │      │
│  └────────────────────────────────────────┘      │
│                     │ output feeds into          │
│                     ▼                            │
│  ┌────────────────────────────────────────┐      │
│  │  Agent: Data Analyst                  │      │
│  │  Role: "Statistical analyst"          │      │
│  │  Goal: "Extract actionable insights"  │      │
│  │  Tools: [calculate, chart_generator]  │      │
│  └────────────────────────────────────────┘      │
│                     │ output feeds into          │
│                     ▼                            │
│  ┌────────────────────────────────────────┐      │
│  │  Agent: Report Writer                 │      │
│  │  Role: "Technical writer"             │      │
│  │  Goal: "Produce clear reports"        │      │
│  │  Tools: [format_document]             │      │
│  └────────────────────────────────────────┘      │
└──────────────────────────────────────────────────┘
```

#### Example: Research Crew

```python
"""
A three-agent crew that researches a topic, analyzes data, and writes a report.
Demonstrates: role-based agents, sequential tasks, tool integration.
"""
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

search_tool = SerperDevTool()

# ── Define Agents ─────────────────────────────────────────────────
researcher = Agent(
    role="Senior Research Analyst",
    goal="Uncover cutting-edge developments in {topic}",
    backstory=(
        "You are a veteran analyst at a leading tech think tank. "
        "Known for your ability to find patterns others miss, "
        "you specialize in identifying emerging trends."
    ),
    tools=[search_tool],
    verbose=True,
    memory=True,              # Agent remembers across tasks
    max_iter=5,               # Max ReAct iterations per task
    allow_delegation=False,   # Cannot delegate to other agents
)

analyst = Agent(
    role="Data Analyst",
    goal="Analyze research data and extract key metrics from {topic}",
    backstory=(
        "You are a quantitative analyst with expertise in data "
        "interpretation. You turn raw research into structured insights."
    ),
    verbose=True,
    memory=True,
)

writer = Agent(
    role="Technical Report Writer",
    goal="Create a comprehensive yet accessible report on {topic}",
    backstory=(
        "You are a seasoned technical writer who translates complex "
        "findings into clear, executive-ready reports."
    ),
    verbose=True,
    memory=True,
)

# ── Define Tasks ──────────────────────────────────────────────────
research_task = Task(
    description=(
        "Conduct thorough research on {topic}. Identify key players, "
        "recent breakthroughs, market trends, and potential risks. "
        "Your research should cover at least 5 distinct subtopics."
    ),
    expected_output="A detailed research brief with citations and key data points.",
    agent=researcher,
)

analysis_task = Task(
    description=(
        "Using the research brief, perform a detailed analysis:\n"
        "1. Identify the top 3 trends\n"
        "2. Quantify market impact where possible\n"
        "3. Assess risks and opportunities\n"
        "4. Create a SWOT summary"
    ),
    expected_output="A structured analysis with trends, metrics, and SWOT.",
    agent=analyst,
)

report_task = Task(
    description=(
        "Compile the research and analysis into a professional report. "
        "Include: Executive Summary, Key Findings, Detailed Analysis, "
        "Recommendations, and Risk Assessment. Target audience: C-suite."
    ),
    expected_output="A polished, executive-ready report in markdown format.",
    agent=writer,
    output_file="report.md",  # Save output to file
)

# ── Assemble and Run the Crew ─────────────────────────────────────
crew = Crew(
    agents=[researcher, analyst, writer],
    tasks=[research_task, analysis_task, report_task],
    process=Process.sequential,  # Tasks run in order
    verbose=True,
    memory=True,                 # Shared crew memory
)

result = crew.kickoff(inputs={"topic": "AI agents in enterprise automation"})
print(result)
```

#### CrewAI Process Types

| Process | Description | Use Case |
|---------|-------------|----------|
| `sequential` | Tasks run one after another | Linear pipelines |
| `hierarchical` | A manager agent delegates to workers | Complex projects needing oversight |

#### CrewAI Key Concepts

| Concept | Description |
|---------|-------------|
| **Agent** | An autonomous unit with role, goal, backstory, and tools |
| **Task** | A specific assignment given to an agent |
| **Crew** | A team of agents working together on tasks |
| **Process** | How tasks are executed (sequential or hierarchical) |
| **Memory** | Shared context across agents and tasks |
| **Delegation** | Agents can ask other agents for help |

---

### 4. Microsoft AutoGen

**What**: A framework for building multi-agent **conversations**. Agents communicate through messages, and complex behaviors emerge from structured conversations.

**Best For**: Conversational multi-agent systems, code generation with execution, human-in-the-loop workflows, research and brainstorming.

**Install**:
```bash
pip install autogen-agentchat autogen-ext[openai]
```

#### Mental Model: Agents Talk to Each Other

```
┌─────────────────────────────────────────────────────┐
│           AUTOGEN CONVERSATION                      │
│                                                     │
│  ┌───────────┐     message     ┌───────────────┐   │
│  │ Assistant │ ──────────────▶ │    Critic     │   │
│  │  Agent    │ ◀────────────── │    Agent      │   │
│  └───────────┘    feedback     └───────────────┘   │
│       │                              │              │
│       │ asks for help                │ requests     │
│       ▼                              ▼ verification │
│  ┌───────────┐                ┌───────────────┐    │
│  │   Coder   │                │    Human      │    │
│  │   Agent   │                │    Proxy      │    │
│  └───────────┘                └───────────────┘    │
│                                                     │
│  ┌─────────────────────────────────────────┐        │
│  │          GROUP CHAT MANAGER             │        │
│  │  Decides who speaks next                │        │
│  └─────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────┘
```

#### Example: Multi-Agent Code Review System

```python
"""
AutoGen multi-agent system for collaborative code review.
Demonstrates: group chat, auto-reply, code execution, human-in-the-loop.
"""
import asyncio
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4o")

# ── Define Specialized Agents ─────────────────────────────────────
architect = AssistantAgent(
    name="Architect",
    model_client=model_client,
    system_message=(
        "You are a senior software architect. Review code designs for "
        "scalability, maintainability, and adherence to SOLID principles. "
        "Suggest improvements. Say APPROVE when satisfied."
    ),
)

security_reviewer = AssistantAgent(
    name="SecurityReviewer",
    model_client=model_client,
    system_message=(
        "You are a security specialist. Review code for vulnerabilities: "
        "injection attacks, auth issues, data exposure, insecure defaults. "
        "Flag issues with severity levels. Say APPROVE when satisfied."
    ),
)

performance_reviewer = AssistantAgent(
    name="PerformanceReviewer",
    model_client=model_client,
    system_message=(
        "You are a performance engineer. Review code for efficiency: "
        "time complexity, memory usage, N+1 queries, caching opportunities. "
        "Say APPROVE when satisfied."
    ),
)

# ── Run Group Chat ────────────────────────────────────────────────
async def main():
    termination = TextMentionTermination("APPROVE")

    team = RoundRobinGroupChat(
        [architect, security_reviewer, performance_reviewer],
        termination_condition=termination,
        max_turns=10,
    )

    code_to_review = """
    Please review this Python function:

    def get_user_data(user_id):
        query = f"SELECT * FROM users WHERE id = {user_id}"
        result = db.execute(query)
        return jsonify(result.fetchall())
    """

    await Console(team.run_stream(task=code_to_review))

asyncio.run(main())
```

#### AutoGen Key Concepts

| Concept | Description |
|---------|-------------|
| `AssistantAgent` | An LLM-powered agent with a system message |
| `UserProxyAgent` | Represents a human — can auto-reply or prompt for input |
| `GroupChat` | A multi-agent conversation container |
| `RoundRobinGroupChat` | Agents take turns speaking |
| `SelectorGroupChat` | An LLM decides who speaks next |
| Termination conditions | Stop when a keyword appears or max turns reached |

---

### Framework Comparison

| Feature | LangGraph | Semantic Kernel | CrewAI | AutoGen |
|---------|-----------|----------------|--------|---------|
| **Paradigm** | Graph-based workflows | Plugin architecture | Role-based crews | Conversational agents |
| **State Model** | Typed state dict | Kernel + memory | Shared crew memory | Message history |
| **Orchestration** | Conditional edges / routing | Planner / auto-invoke | Sequential / hierarchical | Group chat manager |
| **Cycles / Loops** | ✅ First-class | Via planner | Via delegation | Via conversation flow |
| **Human-in-the-Loop** | ✅ Built-in | ✅ Supported | ⚠️ Limited | ✅ UserProxyAgent |
| **Persistence** | ✅ Checkpointing | ✅ Memory stores | ✅ Memory system | ⚠️ External |
| **Code Execution** | Via tools | Via plugins | Via tools | ✅ Built-in sandbox |
| **Best Language** | Python | Python / C# / Java | Python | Python |
| **Learning Curve** | Medium–High | Medium | Low–Medium | Medium |
| **When to Choose** | Complex branching workflows | Enterprise / .NET integration | Quick multi-agent prototypes | Conversational AI research |

---

## 🤝 Multi-Agent Collaboration Patterns

### Pattern 1: Supervisor (Hub-and-Spoke)

A central **supervisor agent** delegates tasks to specialized worker agents.

```
                    ┌──────────────┐
                    │  SUPERVISOR  │
                    │  (decides    │
                    │   who works) │
                    └──────┬───────┘
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ Worker A │ │ Worker B │ │ Worker C │
        │(Research)│ │(Analysis)│ │(Writing) │
        └──────────┘ └──────────┘ └──────────┘
```

**When to use**: When you need centralized control and clear task delegation.

### Pattern 2: Pipeline (Assembly Line)

Agents are chained so one agent's output feeds the next.

```
Agent A ──▶ Agent B ──▶ Agent C ──▶ Agent D
(Gather)    (Clean)     (Analyze)   (Report)
```

**When to use**: Linear workflows where each step transforms data.

### Pattern 3: Debate (Adversarial)

Agents argue different positions, and a judge agent synthesizes the best answer.

```
┌──────────┐     ┌──────────┐
│ Advocate │     │  Critic  │
│ (argues  │◀───▶│ (argues  │
│  FOR)    │     │  AGAINST)│
└────┬─────┘     └────┬─────┘
     │                │
     └───────┬────────┘
             ▼
       ┌──────────┐
       │  JUDGE   │
       │(synthesize│
       │  answer) │
       └──────────┘
```

**When to use**: Complex decisions where multiple perspectives improve quality.

### Pattern 4: Hierarchical (Manager → Team Leads → Workers)

Multi-level delegation for very complex projects.

```
                 ┌──────────────┐
                 │   MANAGER    │
                 └──────┬───────┘
                        │
              ┌─────────┼─────────┐
              ▼                   ▼
        ┌──────────┐        ┌──────────┐
        │ Team Lead│        │ Team Lead│
        │  (Eng)   │        │ (Design) │
        └────┬─────┘        └────┬─────┘
             │                   │
        ┌────┼────┐         ┌────┼────┐
        ▼    ▼    ▼         ▼    ▼    ▼
       W1   W2   W3        W4   W5   W6
```

**When to use**: Large-scale projects requiring organizational structure.

### Pattern 5: Blackboard (Shared Workspace)

All agents read/write to a shared "blackboard"—a common state that any agent can contribute to. An orchestrator decides which agent acts next based on the blackboard's current state.

```
     ┌──────────────────────────────┐
     │         BLACKBOARD           │
     │  ┌──────────────────────┐   │
     │  │ facts: [...]         │   │
     │  │ hypotheses: [...]    │   │
     │  │ status: "analyzing"  │   │
     │  └──────────────────────┘   │
     └──────────┬───────────────────┘
          ▲     │     ▲
          │     │     │
     ┌────┘     │     └────┐
     │          ▼          │
┌────────┐ ┌────────┐ ┌────────┐
│Agent A │ │Agent B │ │Agent C │
│(read/  │ │(read/  │ │(read/  │
│ write) │ │ write) │ │ write) │
└────────┘ └────────┘ └────────┘
```

**When to use**: Problems where the solution emerges from incremental contributions and agents don't have a fixed order.

---

## 💾 Database Querying & Dynamic Memory

### Querying Databases from Agents

Agents often need to query databases as part of their reasoning. Here's a pattern for safe, structured database access:

```python
"""
Database-aware agent with structured query tools.
Demonstrates: safe SQL execution, result formatting, memory updates.
"""
import sqlite3
import json
from typing import Any

class DatabaseTool:
    """A tool that lets agents query databases safely."""

    def __init__(self, db_path: str):
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row

    def execute_query(self, sql: str, params: tuple = ()) -> list[dict]:
        """
        Execute a READ-ONLY query. Blocks INSERT/UPDATE/DELETE.
        This prevents the agent from accidentally modifying data.
        """
        forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "CREATE"]
        if any(word in sql.upper() for word in forbidden):
            raise PermissionError(
                f"Write operations are not allowed. Use dedicated mutation tools."
            )

        cursor = self.conn.execute(sql, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_schema(self) -> str:
        """Return the database schema for the LLM to understand."""
        cursor = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table'"
        )
        schemas = [row[0] for row in cursor.fetchall() if row[0]]
        return "\n\n".join(schemas)

    def natural_language_query(self, question: str, llm) -> dict:
        """
        Convert natural language to SQL, execute, and return results.
        This is the ReAct pattern applied to database querying.
        """
        schema = self.get_schema()

        # Step 1: THINK — Generate SQL
        sql_response = llm.invoke(
            f"Given this schema:\n{schema}\n\n"
            f"Write a SELECT query to answer: {question}\n"
            f"Return ONLY the SQL, no explanation."
        )
        sql = sql_response.content.strip().strip("```sql").strip("```")

        # Step 2: ACT — Execute query
        try:
            results = self.execute_query(sql)
        except Exception as e:
            return {"error": str(e), "sql": sql}

        # Step 3: OBSERVE — Return structured results
        return {
            "question": question,
            "sql": sql,
            "results": results,
            "row_count": len(results),
        }
```

### Dynamic Memory Management

Agents need different types of memory for different purposes:

```python
"""
A dynamic memory system that agents can read and update during execution.
Supports: short-term, long-term, and episodic memory.
"""
from datetime import datetime
from collections import deque
import json

class AgentMemory:
    """
    Three-tier memory system:
    - Short-term:  Current conversation/task context (fast, limited)
    - Long-term:   Persistent facts and learned knowledge (slow, unlimited)
    - Episodic:    Records of past interactions and outcomes (for learning)
    """

    def __init__(self, max_short_term: int = 50):
        self.short_term: deque[dict] = deque(maxlen=max_short_term)
        self.long_term: dict[str, Any] = {}
        self.episodic: list[dict] = []

    # ── Short-term memory (working memory) ────────────────────────
    def remember(self, content: str, metadata: dict = None):
        """Add to short-term memory."""
        self.short_term.append({
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })

    def get_recent_context(self, n: int = 10) -> list[str]:
        """Retrieve the N most recent memories."""
        return [m["content"] for m in list(self.short_term)[-n:]]

    # ── Long-term memory (knowledge base) ─────────────────────────
    def learn(self, key: str, value: Any):
        """Store a fact in long-term memory."""
        self.long_term[key] = {
            "value": value,
            "learned_at": datetime.now().isoformat(),
            "access_count": 0,
        }

    def recall(self, key: str) -> Any:
        """Retrieve a fact from long-term memory."""
        if key in self.long_term:
            self.long_term[key]["access_count"] += 1
            return self.long_term[key]["value"]
        return None

    # ── Episodic memory (experience log) ──────────────────────────
    def record_episode(self, task: str, actions: list, outcome: str,
                       success: bool):
        """Record a complete interaction for future reference."""
        self.episodic.append({
            "task": task,
            "actions": actions,
            "outcome": outcome,
            "success": success,
            "timestamp": datetime.now().isoformat(),
        })

    def find_similar_episodes(self, task: str) -> list[dict]:
        """Find past episodes similar to the current task."""
        # In production, use embedding similarity search
        return [
            ep for ep in self.episodic
            if any(word in ep["task"].lower() for word in task.lower().split())
        ]

    def get_success_rate(self) -> float:
        """Calculate overall success rate from episodic memory."""
        if not self.episodic:
            return 0.0
        successes = sum(1 for ep in self.episodic if ep["success"])
        return successes / len(self.episodic)

# ── Usage ─────────────────────────────────────────────────────────
memory = AgentMemory()

# Short-term: Track current task
memory.remember("User asked about Q3 sales data")
memory.remember("Queried database — found $2.4M total revenue")

# Long-term: Learn reusable facts
memory.learn("company_fiscal_year_start", "April 1")
memory.learn("primary_database", "postgresql://prod-db:5432/sales")

# Episodic: Record outcomes for learning
memory.record_episode(
    task="Generate Q3 sales report",
    actions=["query_database", "analyze_trends", "format_report"],
    outcome="Successfully generated report with 15 data points",
    success=True,
)
```

---

## 🏗 Production Best Practices

### 1. Error Handling & Recovery

```python
import time
from functools import wraps

def retry_with_backoff(max_retries: int = 3, base_delay: float = 1.0):
    """Decorator for retrying failed tool calls with exponential backoff."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    delay = base_delay * (2 ** attempt)
                    print(f"Attempt {attempt+1} failed: {e}. "
                          f"Retrying in {delay}s...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry_with_backoff(max_retries=3)
def call_llm(messages):
    """LLM call with automatic retry on transient failures."""
    return client.chat.completions.create(
        model="gpt-4o", messages=messages
    )
```

### 2. Guardrails & Safety

```python
def validate_agent_action(action: str, allowed_actions: list[str]) -> bool:
    """Ensure the agent only performs permitted actions."""
    return action in allowed_actions

def cost_guard(max_cost_usd: float = 5.0):
    """Track and limit spending on LLM calls."""
    total_cost = 0.0

    def check_cost(input_tokens: int, output_tokens: int):
        nonlocal total_cost
        # Approximate pricing for GPT-4o
        cost = (input_tokens * 2.5 / 1_000_000) + (output_tokens * 10 / 1_000_000)
        total_cost += cost
        if total_cost > max_cost_usd:
            raise RuntimeError(
                f"Cost limit exceeded: ${total_cost:.4f} > ${max_cost_usd}"
            )
        return total_cost

    return check_cost
```

### 3. Observability & Logging

```python
import logging
import uuid
from datetime import datetime

class AgentTracer:
    """Trace agent execution for debugging and monitoring."""

    def __init__(self):
        self.trace_id = str(uuid.uuid4())[:8]
        self.steps: list[dict] = []
        self.logger = logging.getLogger("agent")

    def log_step(self, step_type: str, content: str, metadata: dict = None):
        entry = {
            "trace_id":  self.trace_id,
            "step":      len(self.steps) + 1,
            "type":      step_type,   # "think", "act", "observe", "error"
            "content":   content,
            "metadata":  metadata or {},
            "timestamp": datetime.now().isoformat(),
        }
        self.steps.append(entry)
        self.logger.info(f"[{self.trace_id}] {step_type}: {content[:100]}")

    def get_summary(self) -> dict:
        return {
            "trace_id":    self.trace_id,
            "total_steps": len(self.steps),
            "step_types":  [s["type"] for s in self.steps],
            "duration_ms": "calculated_from_timestamps",
        }
```

### 4. Testing Multi-Agent Systems

```python
import pytest

class TestResearchCrew:
    """Test suite for multi-agent research system."""

    def test_agent_receives_correct_context(self):
        """Verify state is passed correctly between agents."""
        state = ResearchState(
            research_topic="AI Agents",
            messages=[],
            search_results=["result1", "result2"],
            analysis="",
            report="",
            iteration=0,
        )
        result = analyze_results(state)
        assert "analysis" in result
        assert result["iteration"] == 1

    def test_max_iteration_guard(self):
        """Ensure the system doesn't loop forever."""
        state = ResearchState(iteration=3, analysis="NEED_MORE_DATA")
        decision = should_continue_research(state)
        assert decision == "write"  # Should stop despite wanting more data

    def test_tool_error_handling(self):
        """Verify graceful handling of tool failures."""
        result = search_database("nonexistent_query")
        assert "No results found" in result
```

---

## 🎓 Hands-On Learning Exercises

### Exercise 1: Build a ReAct Agent from Scratch *(Beginner)*
**Goal**: Implement the ReAct loop without any framework.
- Create an agent that can answer questions using a calculator and a dictionary lookup tool.
- Implement the Thought → Action → Observation loop manually.
- Add a maximum step limit to prevent infinite loops.

### Exercise 2: LangGraph Workflow with Cycles *(Intermediate)*
**Goal**: Build a content generation pipeline with quality checks.
- Create nodes: `generate_draft`, `review_quality`, `revise_draft`, `publish`.
- Add a conditional edge from `review_quality` that either loops back to `revise_draft` or proceeds to `publish`.
- Add checkpointing so you can resume interrupted workflows.

### Exercise 3: CrewAI Business Process *(Intermediate)*
**Goal**: Automate a customer support workflow.
- **Agent 1 (Classifier)**: Categorize incoming tickets (bug, feature, question).
- **Agent 2 (Researcher)**: Look up relevant documentation.
- **Agent 3 (Responder)**: Draft a response.
- Use hierarchical process with a manager agent.

### Exercise 4: AutoGen Debate System *(Advanced)*
**Goal**: Build two agents that debate a topic.
- Create a `ProAgent` and `ConAgent` with opposing views.
- Add a `JudgeAgent` that evaluates arguments and declares a winner.
- Implement a `SelectorGroupChat` where the judge decides who speaks next.

### Exercise 5: Full Production System *(Advanced)*
**Goal**: Build a complete stateful multi-agent system with persistence.
- Use LangGraph for orchestration.
- Implement database querying tools.
- Add the three-tier memory system.
- Include error handling, retries, cost guards, and tracing.
- Write integration tests.

---

## 🚀 Demo Project: Multi-Agent Research System

This repository includes a **fully runnable demo project** that puts all the concepts above into practice. It implements a multi-agent research pipeline using LangGraph with shared three-tier memory.

### Architecture

```
START → 🔍 Research Agent → 📊 Analysis Agent → ✍️ Writer Agent → ⭐ Quality Reviewer
           ▲                                                              │
           └──────────────── (if quality < 7/10) ─────────────────────────┘
                              (if quality ≥ 7/10) → END ✅
```

### Project Files

| File | Purpose |
|------|---------|
| `memory.py` | Three-tier memory system (short-term, long-term, episodic) |
| `tools.py` | Simulated tools — web search, database query, calculator |
| `state.py` | LangGraph `TypedDict` state with `add_messages` reducer |
| `agents.py` | Four agent node functions + routing logic |
| `workflow.py` | LangGraph `StateGraph` construction with conditional edges |
| `main.py` | CLI entry point with real-time progress display |

### Quick Start

```bash
# 1. Install dependencies
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# 2. Run in dry-run mode (NO API key needed!)
python main.py --topic "AI agents in healthcare" --dry-run
```

### What You'll See

The pipeline executes 4 agents in sequence, displays real-time progress with memory tracking, and saves a polished markdown report:

```
  🔍  Agent: RESEARCH
      ↳ Action: search_web('AI agents in healthcare')
      ↳ Memory: 7 short-term, 3 long-term, 0 episodic

  📊  Agent: ANALYSIS
      ↳ Action: calculate('2500000 / 4')
      ↳ Memory: 9 short-term, 6 long-term, 0 episodic

  ✍️  Agent: WRITER
      ↳ Action: write_report
      ↳ Memory: 11 short-term, 6 long-term, 0 episodic

  ⭐  Agent: REVIEWER
      ↳ Quality: 8.5/10 — ✅ APPROVED (iteration 1)
      ↳ Memory: 12 short-term, 6 long-term, 1 episodic
```

### Key Concepts Demonstrated

1. **Shared State**: All agents read/write to a `WorkflowState` TypedDict
2. **Three-Tier Memory**: Short-term (working context), long-term (learned facts), episodic (experience log)
3. **Feedback Loop**: The quality reviewer can loop back to the research agent
4. **Tool Use**: Agents call search, database, and calculator tools
5. **Traceability**: Complete action log and memory state are tracked

> Every line of code is heavily commented — read the source files for a deep understanding of each pattern.

---

## 📖 Glossary

| Term | Definition |
|------|-----------|
| **Agent** | An autonomous unit that perceives, reasons, and acts to achieve goals |
| **State** | Data that persists across an agent's reasoning steps |
| **ReAct** | Reasoning and Acting — a loop of Think → Act → Observe |
| **Tool** | A function the LLM can invoke to interact with the world |
| **Orchestration** | Coordination of multiple agents to complete complex tasks |
| **Checkpoint** | A saved snapshot of state for resumption or recovery |
| **Working Memory** | Short-lived context for the current task |
| **Long-Term Memory** | Persistent knowledge that survives across sessions |
| **Episodic Memory** | Records of past actions and outcomes used for learning |
| **Graph** | A network of nodes (functions) and edges (transitions) |
| **Plugin** | A modular, reusable capability exposed to an LLM (Semantic Kernel) |
| **Crew** | A team of role-playing agents (CrewAI) |
| **Group Chat** | A multi-agent conversation (AutoGen) |
| **Human-in-the-Loop** | A pattern where a human approves or corrects agent actions |
| **Guardrail** | A safety mechanism that constrains agent behavior |

---

## 📚 Further Reading

- [ReAct Paper](https://arxiv.org/abs/2210.03629) — *Yao et al., 2022*
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [Semantic Kernel Docs](https://learn.microsoft.com/semantic-kernel/overview/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [AutoGen Documentation](https://microsoft.github.io/autogen/)
- [LLM Agents Survey](https://arxiv.org/abs/2308.11432) — Comprehensive survey of LLM-based agents

---

> **💡 Remember**: The best way to learn is to **build**. Start with Exercise 1 (a raw ReAct loop), then progressively adopt frameworks. Each framework is just a structured way to implement the same core patterns you've learned here.
