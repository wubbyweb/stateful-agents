"""
============================================================================
tools.py — Agent Tools (Simulated External Services)
============================================================================

PURPOSE:
    This module defines the "tools" that agents can use to interact with the
    outside world. In a real system, these would call actual APIs, databases,
    or services. Here, they return simulated data so the project runs without
    any external dependencies (perfect for learning and testing).

WHY SIMULATED?
    - You can run this project WITHOUT an API key (--dry-run mode)
    - The behavior is deterministic and reproducible
    - You can study the agent's reasoning without worrying about API costs
    - Easy to swap in real implementations later

ARCHITECTURE:
    Each tool follows the same pattern:
    1. Validate inputs (safety checks)
    2. Execute the operation (simulated or real)
    3. Return a structured string result

    ┌──────────────────────────────────────────────────┐
    │                  AGENT                           │
    │                                                  │
    │    "I need to search for AI trends"              │
    │         │                                        │
    │         ▼                                        │
    │    ┌──────────┐     ┌──────────────────┐         │
    │    │ TOOL     │────▶│ External Service │         │
    │    │ search   │◀────│ (simulated here) │         │
    │    │ _web()   │     └──────────────────┘         │
    │    └──────────┘                                  │
    │         │                                        │
    │         ▼                                        │
    │    "Found 3 results about AI trends..."          │
    └──────────────────────────────────────────────────┘

============================================================================
"""

import json
from datetime import datetime


# ============================================================================
# SIMULATED DATA
# ============================================================================
# These dictionaries simulate external data sources. In a real system,
# you would replace these with actual API calls. The simulated data is
# intentionally rich enough to produce meaningful agent behavior.
# ============================================================================

# ── Simulated web search results ──
# Organized by topic keyword so the agent gets relevant results based
# on what it searches for. Each result has a title, snippet, and source
# to mimic real search engine output.
SIMULATED_SEARCH_DATA = {
    "ai agents": [
        {
            "title": "The Rise of Agentic AI in 2025",
            "snippet": "Agentic AI systems that can autonomously plan, reason, and "
                       "execute tasks are becoming the dominant paradigm. Companies "
                       "report 40% productivity gains from AI agent deployment.",
            "source": "techreview.com",
            "date": "2025-11-15"
        },
        {
            "title": "Multi-Agent Systems in Enterprise",
            "snippet": "Enterprise adoption of multi-agent architectures has grown "
                       "300% year-over-year. Key use cases include customer support, "
                       "data analysis, and automated reporting.",
            "source": "enterprise-ai.org",
            "date": "2025-10-22"
        },
        {
            "title": "Agent Memory and State Management",
            "snippet": "Modern AI agents use three-tier memory systems: working memory "
                       "for current tasks, long-term memory for persistent knowledge, "
                       "and episodic memory for learning from past experiences.",
            "source": "arxiv.org",
            "date": "2025-09-30"
        },
    ],
    "healthcare": [
        {
            "title": "AI Agents Transform Clinical Decision Support",
            "snippet": "Hospital systems using AI agents for diagnostic assistance "
                       "report 25% reduction in diagnostic errors. Multi-agent "
                       "systems coordinate between imaging, lab, and patient history.",
            "source": "healthtech-journal.com",
            "date": "2025-11-01"
        },
        {
            "title": "Drug Discovery Accelerated by Agent Swarms",
            "snippet": "Pharmaceutical companies deploy swarms of specialized agents — "
                       "one for molecule screening, one for toxicity prediction, one "
                       "for clinical trial matching — cutting discovery time by 60%.",
            "source": "pharma-innovation.net",
            "date": "2025-10-15"
        },
    ],
    "productivity": [
        {
            "title": "Developer Productivity with AI Coding Agents",
            "snippet": "Studies show AI coding agents improve developer productivity "
                       "by 30-55%. Agents that maintain session state and remember "
                       "code context outperform stateless alternatives by 2x.",
            "source": "devtools-weekly.com",
            "date": "2025-11-20"
        },
    ],
    "default": [
        {
            "title": "Comprehensive Guide to AI Agents",
            "snippet": "AI agents are autonomous systems that perceive their "
                       "environment, make decisions, and take actions to achieve "
                       "goals. Modern agents use LLMs as their reasoning engine.",
            "source": "ai-guide.org",
            "date": "2025-10-01"
        },
    ],
}

# ── Simulated database records ──
# Mimics a company database with customers, sales, and metrics.
SIMULATED_DATABASE = {
    "customers": [
        {"id": 1, "name": "Acme Corp",    "status": "active",   "revenue": 450000, "industry": "Manufacturing"},
        {"id": 2, "name": "Globex Inc",    "status": "active",   "revenue": 380000, "industry": "Technology"},
        {"id": 3, "name": "Initech",       "status": "inactive", "revenue": 220000, "industry": "Finance"},
        {"id": 4, "name": "Umbrella Ltd",  "status": "active",   "revenue": 560000, "industry": "Healthcare"},
        {"id": 5, "name": "Wayne Ent",     "status": "active",   "revenue": 890000, "industry": "Technology"},
    ],
    "sales_by_quarter": {
        "Q1": {"total": 1200000, "growth": "12%", "top_product": "Widget Pro"},
        "Q2": {"total": 1800000, "growth": "18%", "top_product": "Agent Suite"},
        "Q3": {"total": 2400000, "growth": "25%", "top_product": "Agent Suite"},
        "Q4": {"total": 3100000, "growth": "30%", "top_product": "Agent Enterprise"},
    },
    "metrics": {
        "total_customers": 5,
        "active_customers": 4,
        "total_revenue": 2500000,
        "avg_deal_size": 500000,
        "customer_satisfaction": 4.2,
    },
}


# ============================================================================
# TOOL FUNCTIONS
# ============================================================================
# Each function below is a "tool" that agents can invoke during their ReAct
# reasoning loop. Tools are the bridge between the agent's "thinking" and
# the real world. Without tools, agents can only generate text — with tools,
# they can gather data, perform calculations, and take actions.
# ============================================================================

def search_web(query: str) -> str:
    """
    Simulate a web search and return structured results.

    HOW THIS TOOL IS USED BY AGENTS:
        The Research Agent calls this tool when it needs to gather
        information about a topic. The agent constructs a search query
        based on its understanding of the research topic, and this tool
        returns relevant results.

    HOW THE SIMULATION WORKS:
        We match keywords in the query against our simulated search data.
        If a keyword matches a topic in our data, we return those results.
        Otherwise, we return default results. This ensures the agent always
        gets meaningful data to work with.

    IN PRODUCTION:
        Replace this with a real search API call:
        - Google Custom Search API
        - SerpAPI
        - Tavily Search API (popular for AI agents)

    Args:
        query: The search query string (e.g., "AI agents in healthcare").

    Returns:
        A JSON-formatted string containing search results with title,
        snippet, source, and date for each result.
    """
    # ── Keyword matching against simulated data ──
    # We check if any topic keyword appears in the query. This is
    # intentionally simple — real search engines use complex ranking.
    query_lower = query.lower()
    results = []

    for topic, topic_results in SIMULATED_SEARCH_DATA.items():
        if topic == "default":
            continue
        # Check if the topic keyword appears anywhere in the query
        if topic in query_lower:
            results.extend(topic_results)

    # ── Fallback to default results ──
    # If no topic keywords matched, return generic results so the
    # agent always has something to work with.
    if not results:
        results = SIMULATED_SEARCH_DATA["default"]

    # ── Format as JSON string ──
    # Agents receive tool outputs as strings. JSON formatting makes
    # it easy for the LLM to parse structured data.
    return json.dumps({
        "query": query,
        "result_count": len(results),
        "results": results,
        "timestamp": datetime.now().isoformat(),
    }, indent=2)


def query_database(query_description: str) -> str:
    """
    Simulate a database query based on a natural language description.

    HOW THIS TOOL IS USED BY AGENTS:
        The Research Agent or Analysis Agent calls this when it needs
        structured business data (customers, sales, metrics). The agent
        describes what data it wants in natural language, and this tool
        returns the matching records.

    SAFETY DESIGN:
        In a real system, you'd want to:
        1. Convert natural language to SQL (using the LLM)
        2. VALIDATE the SQL (block INSERT/UPDATE/DELETE)
        3. Execute against a READ-ONLY database replica
        4. Limit result set size
        Here we simulate this with keyword-based routing.

    Args:
        query_description: Natural language description of the data needed.
                          Examples: "active customers", "Q3 sales", "metrics"

    Returns:
        A JSON-formatted string with the query results.
    """
    query_lower = query_description.lower()
    result = {}

    # ── Route to the appropriate simulated data table ──
    # In a real system, this routing would be done by the LLM generating
    # SQL from the natural language query, then executing it.
    if "customer" in query_lower:
        # Filter by status if specified
        if "active" in query_lower:
            records = [c for c in SIMULATED_DATABASE["customers"]
                       if c["status"] == "active"]
        elif "inactive" in query_lower:
            records = [c for c in SIMULATED_DATABASE["customers"]
                       if c["status"] == "inactive"]
        else:
            records = SIMULATED_DATABASE["customers"]

        result = {
            "table": "customers",
            "query": query_description,
            "row_count": len(records),
            "data": records,
        }

    elif "sales" in query_lower or "quarter" in query_lower or "revenue" in query_lower:
        # Check for specific quarter
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            if q.lower() in query_lower:
                result = {
                    "table": "sales_by_quarter",
                    "query": query_description,
                    "quarter": q,
                    "data": SIMULATED_DATABASE["sales_by_quarter"][q],
                }
                break
        else:
            # Return all quarters if no specific one requested
            result = {
                "table": "sales_by_quarter",
                "query": query_description,
                "data": SIMULATED_DATABASE["sales_by_quarter"],
            }

    elif "metric" in query_lower or "summary" in query_lower or "overview" in query_lower:
        result = {
            "table": "metrics",
            "query": query_description,
            "data": SIMULATED_DATABASE["metrics"],
        }

    else:
        # ── No match — return all available tables ──
        # This helps the agent understand what data is available
        # so it can refine its query on the next iteration.
        result = {
            "query": query_description,
            "error": "No matching data found",
            "available_tables": ["customers", "sales_by_quarter", "metrics"],
            "hint": "Try querying for 'customers', 'sales', or 'metrics'",
        }

    return json.dumps(result, indent=2)


def calculate(expression: str) -> str:
    """
    Safely evaluate a mathematical expression.

    HOW THIS TOOL IS USED BY AGENTS:
        The Analysis Agent uses this to perform calculations on data —
        computing growth rates, averages, percentages, etc. The agent
        constructs a math expression based on data it has gathered.

    SAFETY DESIGN:
        We use Python's eval() but with a RESTRICTED namespace that only
        allows mathematical operations. This prevents code injection:
        - ✅ Allowed: "2 + 2", "1500000 * 0.25", "round(3.14159, 2)"
        - ❌ Blocked: "import os; os.system('rm -rf /')"
        - ❌ Blocked: "__import__('subprocess').call(['ls'])"

    WARNING:
        Even with restrictions, eval() has risks in production. Consider
        using a proper math parser (e.g., sympy) or a sandboxed executor
        for production deployments.

    Args:
        expression: A mathematical expression as a string.

    Returns:
        The result of the calculation as a string, or an error message.
    """
    # ── Define a safe namespace for eval ──
    # By setting __builtins__ to a whitelist of math functions, we
    # prevent access to dangerous built-ins like __import__, exec, etc.
    safe_builtins = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "int": int,
        "float": float,
        "pow": pow,
    }

    try:
        # ── Execute in restricted environment ──
        # The second argument to eval() is the globals dict, and the
        # third is the locals dict. By only providing safe_builtins,
        # we prevent access to os, sys, subprocess, etc.
        result = eval(expression, {"__builtins__": safe_builtins})
        return json.dumps({
            "expression": expression,
            "result": result,
            "type": type(result).__name__,
        })
    except Exception as e:
        return json.dumps({
            "expression": expression,
            "error": str(e),
            "hint": "Ensure the expression uses only basic math operations.",
        })


def format_as_report_section(title: str, content: str,
                              section_type: str = "body") -> str:
    """
    Format content into a structured report section.

    HOW THIS TOOL IS USED BY AGENTS:
        The Writer Agent uses this to format its analysis into clean,
        structured sections. Each section gets a consistent format with
        headers, separators, and metadata.

    Args:
        title:        The section title (e.g., "Executive Summary").
        content:      The section body text.
        section_type: One of "header", "body", "conclusion", "data".

    Returns:
        A formatted markdown section string.
    """
    # ── Build the section based on type ──
    # Different section types get different formatting to make the
    # final report visually structured and easy to navigate.
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    if section_type == "header":
        return f"# {title}\n\n*Generated: {timestamp}*\n\n{content}\n"
    elif section_type == "conclusion":
        return f"## {title}\n\n{content}\n\n---\n*Report generated by Multi-Agent Research System*\n"
    elif section_type == "data":
        return f"## {title}\n\n```\n{content}\n```\n"
    else:  # body
        return f"## {title}\n\n{content}\n"


# ============================================================================
# TOOL REGISTRY
# ============================================================================
# This dictionary maps tool names to their functions. The agent nodes
# use this registry to look up and call tools by name. This pattern
# makes it easy to add new tools without modifying agent code.
#
# In LangGraph, tools can also be registered as LangChain Tools using
# the @tool decorator, but we use a simple registry here for clarity.
# ============================================================================

TOOL_REGISTRY = {
    "search_web": {
        "function": search_web,
        "description": "Search the web for information on a topic. "
                      "Input: a search query string.",
    },
    "query_database": {
        "function": query_database,
        "description": "Query the business database for customer, sales, "
                      "or metrics data. Input: a natural language description "
                      "of the data you need.",
    },
    "calculate": {
        "function": calculate,
        "description": "Evaluate a mathematical expression safely. "
                      "Input: a math expression like '2 + 2' or '1500 * 0.3'.",
    },
    "format_as_report_section": {
        "function": format_as_report_section,
        "description": "Format text into a structured report section. "
                      "Input: title, content, and section_type.",
    },
}


def get_tool_descriptions() -> str:
    """
    Generate a formatted string listing all available tools.

    This is injected into agent system prompts so the LLM knows what
    tools are available and how to use them. Clear tool descriptions
    are CRITICAL for good agent behavior — the LLM can only use tools
    it understands.

    Returns:
        A formatted string with tool names and descriptions.
    """
    lines = ["Available Tools:"]
    for name, info in TOOL_REGISTRY.items():
        lines.append(f"  • {name}: {info['description']}")
    return "\n".join(lines)


def execute_tool(tool_name: str, **kwargs) -> str:
    """
    Execute a tool by name with the given arguments.

    This is the central dispatch function that agents call to use tools.
    It handles:
    1. Looking up the tool in the registry
    2. Calling it with the provided arguments
    3. Catching and formatting any errors

    Args:
        tool_name: Name of the tool to execute (must be in TOOL_REGISTRY).
        **kwargs:  Arguments to pass to the tool function.

    Returns:
        The tool's output as a string, or an error message.
    """
    if tool_name not in TOOL_REGISTRY:
        return json.dumps({
            "error": f"Unknown tool: {tool_name}",
            "available_tools": list(TOOL_REGISTRY.keys()),
        })

    try:
        tool_func = TOOL_REGISTRY[tool_name]["function"]
        return tool_func(**kwargs)
    except Exception as e:
        return json.dumps({
            "tool": tool_name,
            "error": str(e),
            "args": kwargs,
        })
