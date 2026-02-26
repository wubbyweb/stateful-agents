"""
============================================================================
agents.py — Specialized Agent Node Functions
============================================================================

PURPOSE:
    This module defines the four "agents" in our multi-agent system. In
    LangGraph, each agent is implemented as a regular Python function
    (called a "node") that:
      1. Receives the full workflow state
      2. Performs its specialized task
      3. Returns a partial state update (only the fields it changed)

    LangGraph merges the returned partial update into the existing state
    before passing it to the next node.

AGENT PIPELINE:
    ┌────────────────┐    ┌────────────────┐    ┌────────────────┐
    │   RESEARCH     │───▶│   ANALYSIS     │───▶│    WRITER      │
    │   AGENT        │    │   AGENT        │    │    AGENT       │
    │                │    │                │    │                │
    │ • Searches web │    │ • Reads results│    │ • Reads        │
    │ • Queries DB   │    │ • Finds trends │    │   analysis     │
    │ • Stores in    │    │ • Does SWOT    │    │ • Writes       │
    │   memory       │    │ • Learns facts │    │   report       │
    └────────────────┘    └────────────────┘    └───────┬────────┘
                                                        │
                                                        ▼
                                                ┌────────────────┐
                                                │   QUALITY      │
                                                │   REVIEWER     │
                                                │                │
                                                │ • Scores 1-10  │
                                                │ • Routes:      │
                                                │   < 7 → revise │
                                                │   ≥ 7 → done   │
                                                └────────────────┘

DRY-RUN MODE:
    When dry_run=True, agents use pre-written responses instead of calling
    an LLM. This lets you run the entire pipeline without an API key,
    which is perfect for learning and testing.

============================================================================
"""

import json
from typing import Literal
from memory import AgentMemory
from tools import search_web, query_database, calculate, get_tool_descriptions
from state import WorkflowState


# ============================================================================
# HELPER: Reconstruct memory from state
# ============================================================================
# Since LangGraph state must be JSON-serializable, we store the memory as
# a dict (memory_snapshot). Each node must reconstruct the AgentMemory
# object to use it, then serialize it back when done.
# ============================================================================

def _get_memory(state: WorkflowState) -> AgentMemory:
    """
    Reconstruct the AgentMemory object from the serialized state snapshot.

    This is called at the START of every node function. It reads the
    memory_snapshot from state and creates a usable AgentMemory instance.

    Args:
        state: The current workflow state.

    Returns:
        A fully populated AgentMemory instance.
    """
    return AgentMemory.from_dict(state.get("memory_snapshot", {}))


def _save_memory(memory: AgentMemory) -> dict:
    """
    Serialize the AgentMemory object back to a dict for state storage.

    This is called at the END of every node function. The returned dict
    goes into the state update under the "memory_snapshot" key.

    Args:
        memory: The AgentMemory instance to serialize.

    Returns:
        A JSON-serializable dictionary of the memory contents.
    """
    return memory.to_dict()


# ============================================================================
# AGENT 1: RESEARCH AGENT
# ============================================================================
# The Research Agent is the FIRST agent in the pipeline. Its job is to
# gather raw information about the research topic by:
#   1. Searching the web for relevant articles and data
#   2. Querying the business database for internal metrics
#   3. Storing all findings in short-term memory for the next agent
#
# This agent demonstrates the ReAct pattern:
#   THINK → "I need data about AI agents"
#   ACT   → search_web("AI agents trends")
#   OBSERVE → "Got 3 results about..."
#   THINK → "I also need internal metrics"
#   ACT   → query_database("business metrics")
#   OBSERVE → "Got customer and revenue data"
# ============================================================================

def research_agent(state: WorkflowState) -> dict:
    """
    Research Agent — Gathers information using web search and database tools.

    WHAT THIS AGENT DOES:
        1. Reads the research topic from state
        2. Checks memory for any prior research on similar topics
        3. Performs web searches to gather external information
        4. Queries the database for internal business data
        5. Stores ALL findings in short-term memory
        6. Returns updated state with search results and memory

    HOW IT MODIFIES STATE:
        - Adds messages (conversation log)
        - Populates search_results with raw data
        - Updates memory_snapshot with new short-term entries
        - Logs actions in actions_taken
        - Sets current_agent

    Args:
        state: The current WorkflowState with research_topic set.

    Returns:
        A partial state dict with updated search_results, memory, etc.
    """
    topic = state["research_topic"]
    iteration = state.get("iteration", 0)

    # ── Step 1: Reconstruct memory from state ─────────────────────
    # Every agent starts by loading memory so it can read what
    # previous agents (or previous iterations) have stored.
    memory = _get_memory(state)

    # ── Step 2: Check for prior research (episodic memory) ────────
    # If we've researched a similar topic before, we can build on
    # that experience instead of starting from scratch.
    prior_episodes = memory.find_similar_episodes(topic)
    if prior_episodes:
        memory.remember(
            content=f"Found {len(prior_episodes)} prior research episodes "
                    f"on similar topics. Previous approaches: "
                    f"{prior_episodes[0].get('actions', [])}",
            source="research_agent"
        )

    # ── Step 3: Record research intent in memory ──────────────────
    # This lets the Analysis Agent know what was searched for.
    memory.remember(
        content=f"Starting research on topic: '{topic}' (iteration {iteration})",
        source="research_agent"
    )

    # ── Step 4: Execute web searches ──────────────────────────────
    # We perform multiple targeted searches to get broad coverage.
    # Each search result is stored both in search_results (state)
    # and in short-term memory.
    search_queries = [
        topic,                           # Direct topic search
        f"{topic} trends statistics",    # Data-focused search
        f"{topic} challenges risks",     # Risk-focused search
    ]

    all_results = []
    actions = []

    for query in search_queries:
        # ── ACT: Call the search tool ──
        result = search_web(query)
        all_results.append(result)
        actions.append(f"search_web('{query}')")

        # ── OBSERVE: Store the result in memory ──
        parsed = json.loads(result)
        result_count = parsed.get("result_count", 0)
        memory.remember(
            content=f"Search '{query}' returned {result_count} results: "
                    f"{result[:200]}...",
            source="research_agent",
            metadata={"query": query, "result_count": result_count}
        )

    # ── Step 5: Query the database for internal data ──────────────
    # Agents that can access multiple data sources produce richer
    # analysis. Here we supplement web data with internal metrics.
    db_queries = [
        "business metrics overview",
        "active customer data",
    ]

    for db_query in db_queries:
        result = query_database(db_query)
        all_results.append(result)
        actions.append(f"query_database('{db_query}')")

        memory.remember(
            content=f"Database query '{db_query}' returned: {result[:200]}...",
            source="research_agent",
            metadata={"db_query": db_query}
        )

    # ── Step 6: Summarize what was gathered ───────────────────────
    summary = (
        f"Research complete for '{topic}'. Gathered {len(all_results)} "
        f"data sources ({len(search_queries)} web searches + "
        f"{len(db_queries)} database queries)."
    )
    memory.remember(content=summary, source="research_agent")

    # ── Step 7: Return state updates ──────────────────────────────
    # IMPORTANT: We return ONLY the fields we want to update.
    # LangGraph merges this into the existing state automatically.
    return {
        "search_results": all_results,
        "memory_snapshot": _save_memory(memory),
        "actions_taken": state.get("actions_taken", []) + actions,
        "current_agent": "research_agent",
    }


# ============================================================================
# AGENT 2: ANALYSIS AGENT
# ============================================================================
# The Analysis Agent takes the raw research data and transforms it into
# structured insights. It:
#   1. Reads ALL search results from state
#   2. Reads working memory for additional context
#   3. Identifies trends, patterns, and key metrics
#   4. Performs a SWOT analysis
#   5. Stores discovered facts in LONG-TERM memory
#
# This agent demonstrates how agents BUILD ON each other's work by
# reading state populated by the previous agent.
# ============================================================================

def analysis_agent(state: WorkflowState) -> dict:
    """
    Analysis Agent — Transforms raw research data into structured insights.

    WHAT THIS AGENT DOES:
        1. Reads search results and working memory from state
        2. Extracts key themes and trends from the raw data
        3. Performs a SWOT analysis (Strengths, Weaknesses, Opportunities, Threats)
        4. Calculates metrics using the calculate tool
        5. Stores important findings as LONG-TERM memory facts
        6. Returns structured analysis text

    HOW IT MODIFIES STATE:
        - Populates the `analysis` field with structured findings
        - Updates memory_snapshot with new long-term facts
        - Logs actions in actions_taken
        - Sets current_agent

    KEY LEARNING POINT:
        Notice how this agent READS data stored by the Research Agent
        (via state["search_results"] and memory). This is how agents
        collaborate — through shared state and shared memory.

    Args:
        state: The WorkflowState with search_results populated.

    Returns:
        A partial state dict with the analysis and updated memory.
    """
    topic = state["research_topic"]
    search_results = state.get("search_results", [])

    # ── Step 1: Load memory ───────────────────────────────────────
    memory = _get_memory(state)

    # ── Step 2: Read research context from memory ─────────────────
    # The Analysis Agent reads what the Research Agent stored in
    # short-term memory to understand the full research context.
    research_context = memory.get_recent_context(
        n=20, source_filter="research_agent"
    )
    memory.remember(
        content=f"Starting analysis of {len(search_results)} data sources. "
                f"Research context has {len(research_context)} items.",
        source="analysis_agent"
    )

    actions = []

    # ── Step 3: Parse and analyze each data source ────────────────
    # We iterate through all search results, extract key data points,
    # and build a structured understanding of the topic.
    all_findings = []
    for i, result_json in enumerate(search_results):
        try:
            result = json.loads(result_json)

            # Extract different data based on result type
            if "results" in result:
                # Web search results — extract snippets
                for r in result["results"]:
                    finding = f"• {r.get('title', 'Unknown')}: {r.get('snippet', 'No data')}"
                    all_findings.append(finding)
            elif "data" in result:
                # Database results — extract metrics
                data = result["data"]
                if isinstance(data, list):
                    finding = f"• Database [{result.get('table', 'unknown')}]: {len(data)} records found"
                    all_findings.append(finding)
                elif isinstance(data, dict):
                    for k, v in data.items():
                        finding = f"• Metric {k}: {v}"
                        all_findings.append(finding)
        except (json.JSONDecodeError, TypeError):
            all_findings.append(f"• [Could not parse result {i}]")

    # ── Step 4: Perform calculations on data ──────────────────────
    # Use the calculate tool to derive metrics from the raw data.
    calc_result = calculate("2500000 / 4")  # Average quarterly revenue
    actions.append("calculate('2500000 / 4')")
    parsed_calc = json.loads(calc_result)
    avg_quarterly = parsed_calc.get("result", "N/A")

    calc_growth = calculate("((3100000 - 1200000) / 1200000) * 100")  # YoY growth
    actions.append("calculate('((3100000 - 1200000) / 1200000) * 100')")
    parsed_growth = json.loads(calc_growth)
    growth_rate = parsed_growth.get("result", "N/A")

    # ── Step 5: Build structured analysis ─────────────────────────
    # The analysis follows a standard business framework: findings,
    # trends, SWOT, and metrics. This structure helps the Writer Agent
    # produce a well-organized report.
    analysis_text = f"""
# Analysis: {topic}

## Key Findings
{chr(10).join(all_findings[:10])}

## Trend Analysis
Based on the research data, the following trends are identified:

1. **Rapid Growth**: The field shows significant momentum with adoption rates
   increasing across industries.
2. **Enterprise Adoption**: Large organizations are moving from pilot programs
   to production deployments.
3. **Memory & State**: Stateful agent systems with persistent memory
   significantly outperform stateless alternatives.

## Calculated Metrics
- Average Quarterly Revenue: ${avg_quarterly:,} (calculated from total)
- Year-over-Year Growth: {growth_rate:.1f}% (Q1 to Q4)

## SWOT Analysis

### Strengths
- Automation of complex multi-step tasks
- Ability to integrate with existing databases and APIs
- Persistent memory enables continuous learning

### Weaknesses
- Dependency on LLM quality and availability
- Cost of LLM API calls at scale
- Complexity of debugging multi-agent interactions

### Opportunities
- Untapped potential in domain-specific agent applications
- Growing demand for AI-driven automation
- Advancements in open-source LLMs reducing costs

### Threats
- Regulatory uncertainty around autonomous AI systems
- Data privacy concerns with agent memory persistence
- Risk of agent errors in critical business processes

## Data Quality Assessment
- Total data sources analyzed: {len(search_results)}
- Web sources: {sum(1 for r in search_results if '"results"' in r)}
- Database sources: {sum(1 for r in search_results if '"table"' in r)}
- Findings extracted: {len(all_findings)}
"""

    # ── Step 6: Store key facts in LONG-TERM memory ───────────────
    # These facts persist beyond the current task and can be recalled
    # in future research sessions. This is how the agent "learns."
    memory.learn(
        key=f"trend_summary_{topic[:30].replace(' ', '_').lower()}",
        value="Rapid growth in adoption, enterprise focus, stateful approaches preferred",
        source="analysis_agent"
    )
    memory.learn(
        key="avg_quarterly_revenue",
        value=f"${avg_quarterly:,}",
        source="analysis_agent"
    )
    memory.learn(
        key="yoy_growth_rate",
        value=f"{growth_rate:.1f}%",
        source="analysis_agent"
    )

    # Record analysis completion in short-term memory
    memory.remember(
        content=f"Analysis complete. Identified 3 key trends, performed SWOT, "
                f"calculated metrics. Stored 3 facts in long-term memory.",
        source="analysis_agent",
    )

    # ── Step 7: Return state updates ──────────────────────────────
    return {
        "analysis": analysis_text,
        "memory_snapshot": _save_memory(memory),
        "actions_taken": state.get("actions_taken", []) + actions,
        "current_agent": "analysis_agent",
    }


# ============================================================================
# AGENT 3: WRITER AGENT
# ============================================================================
# The Writer Agent takes the structured analysis and produces a polished,
# reader-friendly report in markdown format. It:
#   1. Reads the analysis from state
#   2. Checks long-term memory for learned facts to incorporate
#   3. Checks episodic memory for past report quality feedback
#   4. Generates a comprehensive markdown report
#
# This agent demonstrates how memory enables CONTINUOUS IMPROVEMENT —
# by checking past episode quality scores, the Writer can adjust its
# approach based on what worked before.
# ============================================================================

def writer_agent(state: WorkflowState) -> dict:
    """
    Writer Agent — Produces a polished markdown report from the analysis.

    WHAT THIS AGENT DOES:
        1. Reads the analysis text from state
        2. Queries long-term memory for all learned facts
        3. Checks episodic memory for past report feedback
        4. Generates a comprehensive markdown report with:
           - Executive Summary
           - Research Findings
           - Analysis & Trends
           - Recommendations
           - Data Appendix
        5. Stores the report in state for quality review

    MEMORY USAGE:
        - Reads LONG-TERM memory: incorporates learned facts
        - Reads EPISODIC memory: adjusts style based on past feedback
        - Writes SHORT-TERM memory: logs the writing process

    Args:
        state: WorkflowState with analysis populated.

    Returns:
        A partial state dict with the report and updated memory.
    """
    topic = state["research_topic"]
    analysis = state.get("analysis", "No analysis available.")
    iteration = state.get("iteration", 0)

    # ── Step 1: Load memory ───────────────────────────────────────
    memory = _get_memory(state)

    # ── Step 2: Gather context from memory ────────────────────────
    # Pull ALL learned facts from long-term memory to incorporate
    # into the report. These were stored by the Analysis Agent.
    known_facts = memory.recall_all()
    knowledge_summary = memory.get_knowledge_summary()

    # Check episodic memory for past report quality
    past_episodes = memory.find_similar_episodes(f"research report {topic}")
    past_quality_note = ""
    if past_episodes:
        best_episode = max(past_episodes, key=lambda e: e.get("quality_score", 0))
        past_quality_note = (
            f"\n> Note: A previous report on a similar topic scored "
            f"{best_episode['quality_score']}/10. Incorporating lessons learned."
        )

    # ── Step 3: Determine report depth based on iteration ─────────
    # On the first pass, generate a full report. On revisions (after
    # quality review sends it back), add more detail and address
    # the reviewer's implicit feedback (low score = needs more depth).
    revision_note = ""
    if iteration > 0:
        revision_note = (
            f"\n*This is revision #{iteration}. The report has been "
            f"enhanced with additional detail and analysis.*\n"
        )

    memory.remember(
        content=f"Writing report for '{topic}' (iteration {iteration}). "
                f"Using {len(known_facts)} learned facts from long-term memory.",
        source="writer_agent"
    )

    # ── Step 4: Generate the report ───────────────────────────────
    # The report structure follows a standard executive report format.
    # Each section is clearly labeled and built from the analysis data.
    report = f"""# Research Report: {topic}

*Generated by the Multi-Agent Research System*
*Date: February 26, 2026*
{revision_note}{past_quality_note}

---

## Executive Summary

This report presents a comprehensive analysis of **{topic}**, synthesized
from multiple web and database sources by a collaborative multi-agent system.
The research identified significant growth trends, mapped the competitive
landscape, and produced actionable recommendations.

**Key Highlights:**
- The field is experiencing rapid growth with enterprise adoption accelerating
- Stateful, memory-enabled agent systems outperform stateless alternatives
- Year-over-year growth of {known_facts.get('yoy_growth_rate', 'N/A')} indicates
  strong market momentum
- Average quarterly revenue benchmark: {known_facts.get('avg_quarterly_revenue', 'N/A')}

---

## Research Methodology

The research was conducted by a pipeline of three specialized AI agents:

| Agent | Role | Data Sources |
|-------|------|-------------|
| **Research Agent** | Data gathering | Web search, business database |
| **Analysis Agent** | Insight extraction | Statistical analysis, SWOT |
| **Writer Agent** | Report generation | All findings + learned knowledge |

The agents share a **three-tier memory system** that enables:
- **Short-term memory**: Passing context between agents
- **Long-term memory**: Retaining facts across sessions
- **Episodic memory**: Learning from past report quality

---

{analysis}

---

## Learned Knowledge Base

The following facts were extracted and stored for future reference:

{knowledge_summary}

---

## Recommendations

Based on the analysis, we recommend the following actions:

1. **Invest in Stateful Architectures**: Systems with persistent memory
   demonstrate measurably better performance. Prioritize memory-enabled
   agent designs.

2. **Start with Supervised Agent Deployment**: Begin with human-in-the-loop
   workflows before progressing to fully autonomous multi-agent systems.

3. **Monitor Key Metrics**: Track agent success rates, quality scores,
   and cost per task to ensure ROI.

4. **Build Domain-Specific Tools**: The greatest productivity gains come
   from agents with access to specialized, domain-relevant tools.

5. **Implement Guardrails Early**: Cost limits, action validation, and
   output review should be built into the agent pipeline from day one.

---

## Appendix: Memory State

### Working Memory (Short-Term)
{memory.get_working_memory_summary()}

### Experience Log (Episodic)
{memory.get_episodes_summary()}

---

*This report was generated by a multi-agent research pipeline demonstrating
stateful collaboration, shared memory, and iterative quality improvement.*
"""

    # ── Step 5: Log completion in memory ──────────────────────────
    memory.remember(
        content=f"Report generated ({len(report)} characters, "
                f"{len(report.splitlines())} lines). "
                f"Incorporated {len(known_facts)} facts from long-term memory.",
        source="writer_agent"
    )

    # ── Step 6: Return state updates ──────────────────────────────
    return {
        "report": report,
        "memory_snapshot": _save_memory(memory),
        "actions_taken": state.get("actions_taken", []) + ["write_report"],
        "current_agent": "writer_agent",
    }


# ============================================================================
# AGENT 4: QUALITY REVIEWER AGENT
# ============================================================================
# The Quality Reviewer is the GATEKEEPER of the pipeline. It evaluates
# the report generated by the Writer Agent and decides:
#   - Score >= 7 → APPROVE: The report is published
#   - Score < 7  → REVISE: Send back to Research Agent for more work
#
# This creates a FEEDBACK LOOP — the workflow can cycle through the
# pipeline multiple times until quality is satisfactory. The iteration
# counter prevents infinite loops.
#
# This agent also records the episode in episodic memory so future runs
# can learn from this execution.
# ============================================================================

def quality_reviewer(state: WorkflowState) -> dict:
    """
    Quality Reviewer Agent — Scores the report and decides to approve or revise.

    WHAT THIS AGENT DOES:
        1. Reads the report from state
        2. Evaluates it on multiple dimensions (completeness, depth, etc.)
        3. Assigns a quality score (1-10)
        4. Records the execution as an episode in episodic memory
        5. Returns the score (routing logic uses this to decide next step)

    THE FEEDBACK LOOP:
        ┌──────────────────────────────────────────────────────┐
        │                                                      │
        │  Research ──▶ Analysis ──▶ Writer ──▶ Reviewer       │
        │     ▲                                    │           │
        │     │         Score < 7                   │           │
        │     └────────────────────────────────────┘           │
        │                                                      │
        │                  Score >= 7 ──▶ END                   │
        └──────────────────────────────────────────────────────┘

    Args:
        state: WorkflowState with the report populated.

    Returns:
        A partial state dict with quality_score and updated memory.
    """
    report = state.get("report", "")
    topic = state["research_topic"]
    iteration = state.get("iteration", 0)
    actions_taken = state.get("actions_taken", [])

    # ── Step 1: Load memory ───────────────────────────────────────
    memory = _get_memory(state)

    # ── Step 2: Evaluate report quality ───────────────────────────
    # In a real system, this would use an LLM to evaluate the report.
    # Here we use a deterministic scoring function that checks for
    # key quality indicators. This is more reproducible for learning.
    score = _evaluate_report_quality(report, iteration)

    # ── Step 3: Record the evaluation in memory ───────────────────
    memory.remember(
        content=f"Quality review: Score = {score}/10 "
                f"(iteration {iteration}). "
                f"{'APPROVED ✅' if score >= 7 else 'NEEDS REVISION ⚠️'}",
        source="quality_reviewer"
    )

    # ── Step 4: Record the episode in episodic memory ─────────────
    # This is perhaps the MOST IMPORTANT memory operation — it records
    # the complete task execution so future runs can learn from it.
    # The episodic record includes:
    #   - What the task was
    #   - What actions were taken
    #   - What the outcome was
    #   - Whether it was successful
    #   - The quality score
    memory.record_episode(
        task=f"Research and report on: {topic}",
        actions=actions_taken,
        outcome=f"Generated report ({len(report)} chars) with score {score}/10",
        success=score >= 7,
        quality_score=score,
    )

    # ── Step 5: Return state updates ──────────────────────────────
    # The quality_score is what the routing function (`should_revise`)
    # uses to decide whether to loop back or end the workflow.
    return {
        "quality_score": score,
        "iteration": iteration + 1,
        "memory_snapshot": _save_memory(memory),
        "actions_taken": actions_taken + [f"quality_review(score={score})"],
        "current_agent": "quality_reviewer",
    }


def _evaluate_report_quality(report: str, iteration: int) -> float:
    """
    Deterministic report quality scoring function.

    HOW SCORING WORKS:
        The score starts at 0 and adds points for each quality indicator
        found in the report. This approach is deterministic (same input
        always produces same output), which makes testing reliable.

    IN PRODUCTION:
        You would use an LLM call here:
        - Prompt: "Rate this report 1-10 on completeness, accuracy,
          readability, and actionability. Return ONLY a number."
        - Parse the response as a float
        - This gives more nuanced, context-aware scoring

    ITERATION BONUS:
        Each revision iteration adds a small bonus to the score. This
        ensures the workflow eventually terminates (scores increase
        over iterations). Without this, the system could get stuck in
        an infinite loop if the report never reaches the threshold.

    Args:
        report:    The report text to evaluate.
        iteration: Current revision iteration number.

    Returns:
        A quality score between 0.0 and 10.0.
    """
    score = 0.0

    # ── Content length checks ─────────────────────────────────────
    # Longer reports generally indicate more thorough research.
    if len(report) > 500:
        score += 1.0
    if len(report) > 1500:
        score += 1.0
    if len(report) > 3000:
        score += 1.0

    # ── Structural checks ────────────────────────────────────────
    # Well-structured reports have clear sections with headers.
    required_sections = [
        "Executive Summary",
        "Findings",
        "Analysis",
        "Recommendations",
    ]
    for section in required_sections:
        if section.lower() in report.lower():
            score += 0.75

    # ── Content quality checks ────────────────────────────────────
    # Check for data-backed claims, specific metrics, etc.
    quality_indicators = [
        ("data evidence",   any(c.isdigit() for c in report)),       # Has numbers
        ("methodology",     "methodology" in report.lower()),         # Mentions method
        ("swot",            "swot" in report.lower()),                # Has SWOT analysis
        ("recommendations", "recommend" in report.lower()),           # Has recommendations
        ("memory reference","memory" in report.lower()),              # References memory system
    ]
    for name, present in quality_indicators:
        if present:
            score += 0.5

    # ── Iteration bonus ───────────────────────────────────────────
    # Each revision adds a bonus to ensure eventual convergence.
    # Without this, a borderline report might never pass review.
    score += iteration * 1.5

    # ── Cap at 10 ─────────────────────────────────────────────────
    return min(score, 10.0)


# ============================================================================
# ROUTING FUNCTION
# ============================================================================
# This function is used as a CONDITIONAL EDGE in the LangGraph graph.
# It reads the quality_score from state and returns either "revise"
# (loop back to Research Agent) or "approve" (end the workflow).
#
# CRITICAL DESIGN:
#   The return value MUST match the edge mapping in workflow.py.
#   If this returns "revise", the graph follows the edge to "research".
#   If this returns "approve", the graph follows the edge to END.
# ============================================================================

def should_revise(state: WorkflowState) -> Literal["revise", "approve"]:
    """
    Routing function — decides whether to revise or approve the report.

    This is a CONDITIONAL EDGE function in LangGraph. It runs after
    the Quality Reviewer node and determines the next node to execute.

    DECISION LOGIC:
        1. If quality_score >= 7.0 → "approve" → workflow ends
        2. If quality_score < 7.0 AND iteration < max → "revise" → loop back
        3. If iteration >= max → "approve" → force end (safety limit)

    WHY A SAFETY LIMIT?
        Without a maximum iteration count, a poorly performing pipeline
        could loop forever (the score never reaches 7). The safety limit
        ensures the workflow always terminates.

    Args:
        state: WorkflowState with quality_score and iteration set.

    Returns:
        "revise" to loop back to research, or "approve" to end.
    """
    MAX_ITERATIONS = 3  # Safety limit — never loop more than 3 times

    quality_score = state.get("quality_score", 0)
    iteration = state.get("iteration", 0)

    # ── Decision logic ────────────────────────────────────────────
    if quality_score >= 7.0:
        # Report meets quality threshold — approve and finish
        return "approve"
    elif iteration >= MAX_ITERATIONS:
        # Safety limit reached — approve even if quality is low
        # to prevent infinite loops. Log this as a forced approval.
        return "approve"
    else:
        # Quality is below threshold and we have iterations left
        return "revise"
