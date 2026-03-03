"""
============================================================================
state.py — LangGraph Workflow State Definition
============================================================================

PURPOSE:
    This module defines the shared state object that flows through every
    node (agent) in the LangGraph workflow. Think of it as the "clipboard"
    that each agent reads from and writes to as the task progresses.

HOW LANGGRAPH STATE WORKS:
    1. You define a TypedDict (or dataclass) that represents ALL the data
       your workflow needs to track.
    2. Each node function receives the FULL state as input.
    3. Each node function returns a PARTIAL dict with only the fields it
       wants to update — LangGraph merges this into the existing state.
    4. Special "reducer" annotations (like `add_messages`) control HOW
       fields are merged. For example, `add_messages` APPENDS new messages
       to the list instead of replacing it.

    ┌────────────────────────────────────────────────────────────────┐
    │                    STATE FLOW                                  │
    │                                                                │
    │  Initial State ─▶ Node A updates fields ─▶ Merged State       │
    │                   ─▶ Node B updates fields ─▶ Merged State     │
    │                   ─▶ Node C updates fields ─▶ Final State      │
    │                                                                │
    │  Each node sees the FULL accumulated state from all previous   │
    │  nodes, plus any new updates from the current node.            │
    └────────────────────────────────────────────────────────────────┘

WHY TypedDict?
    - LangGraph requires the state to be a TypedDict (not a dataclass)
      because it needs to merge partial updates into the state.
    - TypedDict provides type hints for IDE autocomplete and type checking
      without adding runtime overhead.
    - Each field's type annotation tells LangGraph how to handle updates.

============================================================================
"""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


class WorkflowState(TypedDict):
    """
    The shared state that flows through the entire multi-agent workflow.

    Every node (agent) in the LangGraph graph receives this state as its
    input and returns a partial dictionary to update specific fields.

    FIELD-BY-FIELD EXPLANATION:

    messages:
        The conversation history, shared across all agents. Uses the
        `add_messages` reducer so new messages are APPENDED to the list
        instead of replacing it. This is critical — without the reducer,
        each node would overwrite all previous messages.

        The `Annotated[list, add_messages]` syntax is LangGraph's way of
        saying: "When a node returns {'messages': [new_msg]}, APPEND
        new_msg to the existing list instead of replacing it."

    research_topic:
        The user's research topic, set once at the start and read by
        all agents. This is the "goal" that drives the entire workflow.

    search_results:
        Raw search results gathered by the Research Agent. The Analysis
        Agent reads these to perform its analysis. Stored as a list of
        strings (JSON-formatted search results).

    analysis:
        The structured analysis produced by the Analysis Agent. Contains
        trends, insights, and SWOT analysis. The Writer Agent reads this
        to create the final report.

    report:
        The final written report produced by the Writer Agent. This is
        the primary output of the entire workflow.

    memory_snapshot:
        A serialized snapshot of the AgentMemory object (from memory.py).
        Since LangGraph state must be JSON-serializable, we can't store
        the AgentMemory object directly — we serialize it to a dict.

        WHY A SNAPSHOT?
        - LangGraph persists state via checkpointing (SQLite/Postgres).
        - The memory must survive serialization/deserialization.
        - Each node reads the snapshot, reconstructs the AgentMemory
          object, uses it, then writes a new snapshot back.

    iteration:
        Counts how many times the workflow has looped (when the quality
        reviewer sends work back for revision). Used as a safety limit
        to prevent infinite loops — if iteration >= max, we force the
        workflow to end.

    quality_score:
        A 1-10 rating assigned by the Quality Reviewer agent. If the
        score is below the threshold (default: 7), the workflow loops
        back for another revision. This creates a self-improving cycle.

    actions_taken:
        A log of every action taken during the workflow. Used for
        traceability, debugging, and episodic memory recording.
        Each entry is a string like "research_agent: searched for X".

    current_agent:
        The name of the agent currently executing. Used for logging
        and display purposes — lets the main.py script show which
        agent is active in the console output.
    """

    # ── Core conversation data ────────────────────────────────────
    # The `Annotated[list, add_messages]` type tells LangGraph to use
    # the add_messages reducer. This means when a node returns:
    #   {"messages": [HumanMessage("hello")]}
    # LangGraph APPENDS that message to the existing list, rather than
    # replacing the entire message list. This is essential for
    # maintaining conversation history across multiple agents.
    messages: Annotated[list, add_messages]

    # ── Research data ─────────────────────────────────────────────
    research_topic: str        # The topic to research (set by user)
    search_results: list[str]  # Raw search/database results (set by Research Agent)

    # ── Analysis & output ─────────────────────────────────────────
    analysis: str              # Structured analysis (set by Analysis Agent)
    report: str                # Final report markdown (set by Writer Agent)

    # ── Memory ────────────────────────────────────────────────────
    # Serialized AgentMemory — see memory.py for the full class.
    # Nodes reconstruct the AgentMemory from this dict, use it, then
    # write the updated dict back.
    memory_snapshot: dict

    # ── Workflow control ──────────────────────────────────────────
    iteration: int             # Current revision loop count (0-based)
    quality_score: float       # Report quality rating (1-10)

    # ── Context isolation ─────────────────────────────────────
    thread_id: str             # Unique execution/conversation identifier

    # ── Observability ─────────────────────────────────────────
    actions_taken: list[str]   # Log of all actions for traceability
    current_agent: str         # Name of the currently executing agent


def create_initial_state(topic: str, memory_snapshot: dict = None,
                         thread_id: str = "") -> dict:
    """
    Create the initial state for a new workflow execution.

    This function is called by main.py to set up the starting state
    before invoking the LangGraph graph. It initializes all fields
    with sensible defaults.

    Args:
        topic:           The research topic provided by the user.
        memory_snapshot: Optional pre-existing memory to load (e.g.,
                        from a previous session). If None, starts fresh.
        thread_id:       Unique execution/conversation ID for context
                        isolation across distributed backends.

    Returns:
        A dictionary matching the WorkflowState schema, ready to be
        passed to `graph.invoke()`.

    Example:
        state = create_initial_state(
            topic="AI agents in healthcare",
            memory_snapshot=None,  # Fresh start
            thread_id="abc-123",
        )
        result = graph.invoke(state)
    """
    return {
        # Messages start empty — agents will populate the conversation
        "messages": [],

        # The user's research topic drives the entire workflow
        "research_topic": topic,

        # No results yet — the Research Agent will populate this
        "search_results": [],

        # No analysis yet — the Analysis Agent will populate this
        "analysis": "",

        # No report yet — the Writer Agent will populate this
        "report": "",

        # Memory starts empty or from a previous session
        "memory_snapshot": memory_snapshot or {
            "short_term": [],
            "long_term": {},
            "episodic": [],
        },

        # Iteration counter starts at 0
        "iteration": 0,

        # No quality score yet — the Reviewer will set this
        "quality_score": 0.0,

        # Unique execution context for distributed backend isolation
        "thread_id": thread_id,

        # Action log starts empty
        "actions_taken": [],

        # No agent is active yet
        "current_agent": "",
    }
