"""
============================================================================
workflow.py — LangGraph Workflow Construction
============================================================================

PURPOSE:
    This module assembles the multi-agent pipeline as a LangGraph StateGraph.
    It connects the four agent nodes with edges (including one conditional
    edge that creates the quality-review feedback loop).

HOW A LANGGRAPH WORKFLOW IS BUILT:
    Building a LangGraph workflow is a 4-step process:

    Step 1: Create a StateGraph with your state type
    Step 2: Add nodes (each node is a function that transforms state)
    Step 3: Add edges (connections between nodes, including conditionals)
    Step 4: Compile the graph into a runnable application

    ┌──────────────────────────────────────────────────────────────┐
    │                    GRAPH STRUCTURE                           │
    │                                                              │
    │  START                                                       │
    │    │                                                         │
    │    ▼                                                         │
    │  ┌────────────┐                                              │
    │  │  research  │ ◀──── (loop back on "revise")               │
    │  └──────┬─────┘                                              │
    │         │                                                    │
    │         ▼                                                    │
    │  ┌────────────┐                                              │
    │  │  analysis  │                                              │
    │  └──────┬─────┘                                              │
    │         │                                                    │
    │         ▼                                                    │
    │  ┌────────────┐                                              │
    │  │   writer   │                                              │
    │  └──────┬─────┘                                              │
    │         │                                                    │
    │         ▼                                                    │
    │  ┌────────────┐                                              │
    │  │  reviewer  │                                              │
    │  └──────┬─────┘                                              │
    │         │                                                    │
    │         ├── score >= 7 ──▶ END  ✅                           │
    │         │                                                    │
    │         └── score < 7 ───▶ research (loop back) ↩️           │
    │                                                              │
    └──────────────────────────────────────────────────────────────┘

CONDITIONAL EDGES — THE KEY TO POWERFUL WORKFLOWS:
    The conditional edge after the "reviewer" node is what makes this
    workflow more than a simple pipeline. It introduces a CYCLE:
    - If the report quality is low, the workflow LOOPS BACK to
      the research agent for another pass.
    - If quality is high enough, the workflow ends.
    This is impossible with simple chains (A→B→C) — you need a graph.

============================================================================
"""

from langgraph.graph import StateGraph, END
from state import WorkflowState
from agents import (
    research_agent,
    analysis_agent,
    writer_agent,
    quality_reviewer,
    should_revise,
)


def build_workflow() -> StateGraph:
    """
    Build and return the compiled LangGraph workflow.

    This function assembles the entire multi-agent pipeline by:
    1. Creating a StateGraph parameterized with our WorkflowState type
    2. Adding four nodes (one per agent)
    3. Connecting them with edges (including one conditional edge)
    4. Compiling into a runnable application

    Returns:
        A compiled LangGraph application ready to be invoked with
        graph.invoke(initial_state).

    Example:
        graph = build_workflow()
        result = graph.invoke(create_initial_state("AI agents"))
        print(result["report"])
    """

    # ══════════════════════════════════════════════════════════════
    # STEP 1: Create the StateGraph
    # ══════════════════════════════════════════════════════════════
    # The StateGraph is parameterized with our WorkflowState TypedDict.
    # This tells LangGraph the shape of the state that flows through
    # every node. LangGraph uses this to:
    #   - Validate that nodes return valid state updates
    #   - Apply reducers (like add_messages) correctly
    #   - Serialize state for checkpointing
    workflow = StateGraph(WorkflowState)

    # ══════════════════════════════════════════════════════════════
    # STEP 2: Add Nodes
    # ══════════════════════════════════════════════════════════════
    # Each node is a function that takes WorkflowState and returns
    # a partial dict of state updates. The string name (first arg)
    # is how we reference the node in edge definitions.
    #
    # NAMING CONVENTION:
    #   Use short, descriptive names. These appear in logs, traces,
    #   and the graph visualization — keep them readable.

    # Node 1: Research Agent — gathers data from web and database
    workflow.add_node("research", research_agent)

    # Node 2: Analysis Agent — transforms raw data into insights
    workflow.add_node("analysis", analysis_agent)

    # Node 3: Writer Agent — generates the markdown report
    workflow.add_node("writer", writer_agent)

    # Node 4: Quality Reviewer — scores the report (1-10)
    workflow.add_node("reviewer", quality_reviewer)

    # ══════════════════════════════════════════════════════════════
    # STEP 3: Add Edges
    # ══════════════════════════════════════════════════════════════
    # Edges define the execution flow. There are three types:
    #
    # 1. ENTRY POINT: Where the graph starts executing
    #    workflow.set_entry_point("node_name")
    #
    # 2. NORMAL EDGE: Unconditional A → B connection
    #    workflow.add_edge("A", "B")
    #
    # 3. CONDITIONAL EDGE: A → (B or C) based on a routing function
    #    workflow.add_conditional_edges("A", router_fn, mapping)

    # ── Entry point ──
    # The workflow always starts with the Research Agent.
    # This is the FIRST node that executes when you call graph.invoke().
    workflow.set_entry_point("research")

    # ── Linear edges (unconditional) ──
    # These define the main pipeline: research → analysis → writer → reviewer
    # After research completes, ALWAYS proceed to analysis.
    workflow.add_edge("research", "analysis")

    # After analysis completes, ALWAYS proceed to writer.
    workflow.add_edge("analysis", "writer")

    # After writer completes, ALWAYS proceed to reviewer.
    workflow.add_edge("writer", "reviewer")

    # ── Conditional edge (the feedback loop) ──
    # This is the KEY FEATURE that makes this a GRAPH, not just a chain.
    #
    # After the reviewer scores the report, the `should_revise` function
    # is called with the current state. It returns either:
    #   "revise"  → the graph loops BACK to "research" for another pass
    #   "approve" → the graph proceeds to END (workflow complete)
    #
    # The mapping dict translates router return values to node names:
    #   {"revise": "research", "approve": END}
    #
    # This creates a CYCLE in the graph — the hallmark of LangGraph
    # that distinguishes it from simple chain-based frameworks.
    workflow.add_conditional_edges(
        "reviewer",          # Source node: runs after "reviewer" completes
        should_revise,       # Router function: reads state, returns "revise" or "approve"
        {
            "revise": "research",  # If "revise" → loop back to research
            "approve": END,        # If "approve" → end the workflow
        }
    )

    # ══════════════════════════════════════════════════════════════
    # STEP 4: Compile the Graph
    # ══════════════════════════════════════════════════════════════
    # Compiling "freezes" the graph definition and produces a runnable
    # application. After compilation, you can:
    #   - Call graph.invoke(state) to run the workflow
    #   - Call graph.stream(state) to stream node-by-node output
    #   - Visualize the graph structure
    #
    # OPTIONAL: Add a checkpointer for persistence:
    #   from langgraph.checkpoint.sqlite import SqliteSaver
    #   memory = SqliteSaver.from_conn_string("checkpoints.db")
    #   compiled = workflow.compile(checkpointer=memory)
    #
    # With a checkpointer, the state is saved after each node, and you
    # can resume interrupted workflows by providing the same thread_id.
    compiled = workflow.compile()

    return compiled


def get_graph_description() -> str:
    """
    Return a human-readable description of the workflow graph.

    This is used by main.py to display the graph structure to the user
    before execution starts. It helps visualize the agent pipeline.

    Returns:
        A multi-line string describing the graph structure.
    """
    return """
╔══════════════════════════════════════════════════════════════╗
║             MULTI-AGENT RESEARCH WORKFLOW                   ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║   START                                                      ║
║     │                                                        ║
║     ▼                                                        ║
║   ┌────────────────────────────────────────────┐             ║
║   │  🔍 RESEARCH AGENT                        │             ║
║   │  • Searches web for topic information      │             ║
║   │  • Queries business database               │◀── LOOP    ║
║   │  • Stores findings in short-term memory    │    BACK     ║
║   └──────────────────┬─────────────────────────┘     │      ║
║                      │                                │      ║
║                      ▼                                │      ║
║   ┌────────────────────────────────────────────┐     │      ║
║   │  📊 ANALYSIS AGENT                        │     │      ║
║   │  • Extracts trends from research data      │     │      ║
║   │  • Performs SWOT analysis                  │     │      ║
║   │  • Stores facts in long-term memory        │     │      ║
║   └──────────────────┬─────────────────────────┘     │      ║
║                      │                                │      ║
║                      ▼                                │      ║
║   ┌────────────────────────────────────────────┐     │      ║
║   │  ✍️  WRITER AGENT                          │     │      ║
║   │  • Reads analysis + learned facts          │     │      ║
║   │  • Checks past report quality (episodic)   │     │      ║
║   │  • Generates markdown report               │     │      ║
║   └──────────────────┬─────────────────────────┘     │      ║
║                      │                                │      ║
║                      ▼                                │      ║
║   ┌────────────────────────────────────────────┐     │      ║
║   │  ⭐ QUALITY REVIEWER                      │     │      ║
║   │  • Scores report quality (1-10)            │     │      ║
║   │  • Records episode in memory               │     │      ║
║   │  • Routes: ≥7 → END, <7 → REVISE          │─────┘      ║
║   └──────────────────┬─────────────────────────┘             ║
║                      │                                        ║
║                      ▼                                        ║
║                    [END]  ✅                                  ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
