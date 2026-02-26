"""
============================================================================
main.py — CLI Entry Point for the Multi-Agent Research System
============================================================================

PURPOSE:
    This is the entry point that ties everything together. It:
    1. Parses command-line arguments (topic, dry-run mode)
    2. Initializes the memory system
    3. Builds the LangGraph workflow
    4. Runs the multi-agent pipeline with real-time progress display
    5. Saves the final report to output/report.md

USAGE:
    # Dry-run mode (no API key needed — uses simulated data):
    python main.py --topic "AI agents in healthcare" --dry-run

    # With a real OpenAI API key:
    export OPENAI_API_KEY=your-key-here
    python main.py --topic "AI agents in healthcare"

    # Interactive mode (prompts for topic):
    python main.py --dry-run

HOW IT WORKS:
    ┌──────────────────────────────────────────────────────────┐
    │                    main.py                                │
    │                                                          │
    │  1. Parse args ─▶ 2. Init memory ─▶ 3. Build graph      │
    │                                          │               │
    │                                          ▼               │
    │  6. Save report ◀── 5. Display ◀── 4. Run graph         │
    │         │                                                │
    │         ▼                                                │
    │    output/report.md                                      │
    └──────────────────────────────────────────────────────────┘

============================================================================
"""

import argparse
import os
import sys
import json
from datetime import datetime

# ── Load environment variables from .env file ─────────────────────
# python-dotenv reads the .env file and sets environment variables.
# This is how we securely load the OpenAI API key without hardcoding it.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional — env vars can be set directly

from memory import AgentMemory
from state import create_initial_state
from workflow import build_workflow, get_graph_description


def print_banner():
    """
    Display a welcome banner with project information.

    This gives the user immediate context about what they're running
    and sets expectations for the output they'll see.
    """
    print("""
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║    🤖  MULTI-AGENT RESEARCH SYSTEM                          ║
║                                                              ║
║    A demonstration of stateful multi-agent collaboration     ║
║    with shared memory using LangGraph.                       ║
║                                                              ║
║    Agents: Research → Analysis → Writer → Quality Review     ║
║    Memory: Short-term + Long-term + Episodic                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)


def print_section(title: str, content: str = "", emoji: str = "📌"):
    """
    Print a formatted section header for console output.

    This provides clear visual separation between different phases
    of execution, making it easy to follow the agent pipeline.

    Args:
        title:   Section title text.
        content: Optional body text to display.
        emoji:   Emoji prefix for visual clarity.
    """
    print(f"\n{'─' * 60}")
    print(f"  {emoji}  {title}")
    print(f"{'─' * 60}")
    if content:
        # Indent content for readability
        for line in content.strip().split("\n"):
            print(f"  {line}")
        print()


def run_pipeline(topic: str, dry_run: bool = True):
    """
    Execute the full multi-agent research pipeline.

    THIS IS THE HEART OF THE APPLICATION. It:
    1. Initializes the shared memory system
    2. Creates the initial workflow state
    3. Builds and runs the LangGraph workflow
    4. Streams progress to the console as each agent executes
    5. Saves the final report

    Args:
        topic:   The research topic to investigate.
        dry_run: If True, uses simulated data (no OpenAI API needed).
                 If False, requires OPENAI_API_KEY environment variable.

    The dry_run flag is key for learning — it lets you study the
    entire multi-agent pipeline without needing an API key or
    incurring any costs.
    """

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: Initialize Memory
    # ══════════════════════════════════════════════════════════════
    # Create a fresh AgentMemory instance. In a production system,
    # you might load a previously saved memory from disk or database
    # to enable cross-session learning.
    print_section("PHASE 1: Initializing Memory System", emoji="🧠")

    memory = AgentMemory(max_short_term=100)

    # ── Seed long-term memory with baseline knowledge ─────────────
    # Pre-loading some facts simulates a system that has learned from
    # prior sessions. The agents can recall these facts during execution.
    memory.learn(
        key="system_version",
        value="Multi-Agent Research System v1.0",
        source="system"
    )
    memory.learn(
        key="preferred_report_format",
        value="Executive summary, then detailed findings, then recommendations",
        source="system"
    )
    memory.learn(
        key="quality_threshold",
        value="Reports must score >= 7/10 to pass review",
        source="system"
    )

    print(f"  ✅ Memory initialized: {memory}")
    print(f"  📝 Pre-loaded {len(memory.long_term)} facts into long-term memory")

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: Create Initial State
    # ══════════════════════════════════════════════════════════════
    # The initial state sets the research topic and injects the
    # serialized memory. This state object will flow through every
    # node in the graph, accumulating data as each agent executes.
    print_section("PHASE 2: Creating Workflow State", emoji="📋")

    initial_state = create_initial_state(
        topic=topic,
        memory_snapshot=memory.to_dict()
    )

    print(f"  📌 Research topic: '{topic}'")
    print(f"  📌 State fields: {list(initial_state.keys())}")
    print(f"  📌 Memory snapshot size: {len(json.dumps(initial_state['memory_snapshot']))} chars")

    # ══════════════════════════════════════════════════════════════
    # PHASE 3: Build the Workflow Graph
    # ══════════════════════════════════════════════════════════════
    # Compile the LangGraph StateGraph into a runnable application.
    # This is where all the nodes and edges defined in workflow.py
    # are assembled into an executable pipeline.
    print_section("PHASE 3: Building Workflow Graph", emoji="🔧")
    print(get_graph_description())

    graph = build_workflow()
    print("  ✅ Workflow graph compiled successfully")

    # ══════════════════════════════════════════════════════════════
    # PHASE 4: Execute the Pipeline
    # ══════════════════════════════════════════════════════════════
    # Run the graph with the initial state. We use graph.invoke()
    # which runs the ENTIRE pipeline synchronously and returns the
    # final state.
    #
    # ALTERNATIVE: graph.stream() returns state after each node,
    # which is useful for showing real-time progress.
    print_section("PHASE 4: Executing Multi-Agent Pipeline", emoji="🚀")
    print("  Agents will execute in order: Research → Analysis → Writer → Reviewer")
    print("  The reviewer may loop back if quality < 7/10\n")

    try:
        # ── Stream node-by-node output ────────────────────────────
        # graph.stream() yields the state update from each node as
        # it completes. This lets us display progress in real-time.
        final_state = None

        for step_output in graph.stream(initial_state):
            # step_output is a dict: {"node_name": {state_updates}}
            for node_name, state_update in step_output.items():
                current_agent = state_update.get("current_agent", node_name)
                actions = state_update.get("actions_taken", [])

                # ── Display agent activity ────────────────────────
                agent_emojis = {
                    "research": "🔍",
                    "analysis": "📊",
                    "writer": "✍️",
                    "reviewer": "⭐",
                }
                emoji = agent_emojis.get(node_name, "🤖")

                print(f"  {emoji}  Agent: {node_name.upper()}")

                # Show actions taken by this agent
                new_actions = actions[-3:]  # Show last 3 actions
                if new_actions:
                    for action in new_actions:
                        print(f"      ↳ Action: {action}")

                # Show memory changes
                mem_snapshot = state_update.get("memory_snapshot", {})
                if mem_snapshot:
                    st_count = len(mem_snapshot.get("short_term", []))
                    lt_count = len(mem_snapshot.get("long_term", {}))
                    ep_count = len(mem_snapshot.get("episodic", []))
                    print(f"      ↳ Memory: {st_count} short-term, "
                          f"{lt_count} long-term, {ep_count} episodic")

                # Show quality score if from reviewer
                if "quality_score" in state_update:
                    score = state_update["quality_score"]
                    iteration = state_update.get("iteration", 0)
                    status = "✅ APPROVED" if score >= 7 else "⚠️ NEEDS REVISION"
                    print(f"      ↳ Quality: {score}/10 — {status} "
                          f"(iteration {iteration})")

                print()

                # Keep track of final state
                if final_state is None:
                    final_state = dict(initial_state)
                final_state.update(state_update)

    except Exception as e:
        print(f"\n  ❌ Pipeline error: {e}")
        print(f"  💡 Make sure you have the required dependencies installed:")
        print(f"     pip install -r requirements.txt")
        if not dry_run:
            print(f"  💡 For dry-run mode (no API key needed): python main.py --dry-run")
        raise

    # ══════════════════════════════════════════════════════════════
    # PHASE 5: Display Results
    # ══════════════════════════════════════════════════════════════
    print_section("PHASE 5: Results", emoji="📊")

    if final_state:
        # ── Quality summary ───────────────────────────────────────
        quality_score = final_state.get("quality_score", 0)
        iterations = final_state.get("iteration", 0)
        total_actions = len(final_state.get("actions_taken", []))
        report = final_state.get("report", "")

        print(f"  📈 Final Quality Score: {quality_score}/10")
        print(f"  🔄 Total Iterations: {iterations}")
        print(f"  ⚡ Total Actions Taken: {total_actions}")
        print(f"  📝 Report Length: {len(report)} characters, "
              f"{len(report.splitlines())} lines")

        # ── Memory final state ────────────────────────────────────
        final_memory = AgentMemory.from_dict(
            final_state.get("memory_snapshot", {})
        )
        print(f"\n  🧠 Final Memory State:")
        print(f"     Short-term: {len(final_memory.short_term)} items")
        print(f"     Long-term:  {len(final_memory.long_term)} facts")
        print(f"     Episodic:   {len(final_memory.episodic)} episodes")
        print(f"     Success Rate: {final_memory.get_success_rate():.0%}")
        print(f"     Avg Quality:  {final_memory.get_average_quality():.1f}/10")

        # ── Action log ────────────────────────────────────────────
        print_section("Action Log", emoji="📋")
        for i, action in enumerate(final_state.get("actions_taken", []), 1):
            print(f"  {i:2d}. {action}")

        # ── Save report ───────────────────────────────────────────
        save_report(report, topic)

        # ── Display report preview ────────────────────────────────
        print_section("Report Preview (first 30 lines)", emoji="📄")
        preview_lines = report.strip().split("\n")[:30]
        for line in preview_lines:
            print(f"  {line}")
        if len(report.strip().split("\n")) > 30:
            print(f"\n  ... ({len(report.splitlines()) - 30} more lines)")
            print(f"  📁 Full report saved to: output/report.md")

    else:
        print("  ⚠️ No final state returned — pipeline may have failed.")


def save_report(report: str, topic: str):
    """
    Save the generated report to the output directory.

    Creates the output directory if it doesn't exist, writes the report
    as a markdown file, and confirms the save to the user.

    Args:
        report: The full markdown report text.
        topic:  The research topic (used in the filename metadata).
    """
    # ── Create output directory ───────────────────────────────────
    # os.makedirs with exist_ok=True is safe to call even if the
    # directory already exists — it won't raise an error.
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
    os.makedirs(output_dir, exist_ok=True)

    # ── Write the report ──────────────────────────────────────────
    output_path = os.path.join(output_dir, "report.md")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print_section(
        f"Report Saved",
        f"File: {output_path}\n"
        f"Size: {len(report):,} characters\n"
        f"Topic: {topic}",
        emoji="💾"
    )


def parse_args():
    """
    Parse command-line arguments.

    Supports:
        --topic     The research topic (optional — prompts if not given)
        --dry-run   Run without OpenAI API key (uses simulated responses)

    Returns:
        An argparse.Namespace with `topic` and `dry_run` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Multi-Agent Research System — Stateful AI Agent Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --topic "AI agents in healthcare" --dry-run
  python main.py --topic "Future of software development"
  python main.py --dry-run   (interactive topic prompt)
        """
    )
    parser.add_argument(
        "--topic",
        type=str,
        default=None,
        help="The research topic to investigate"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Run with simulated data (no OpenAI API key required). "
             "This is the DEFAULT mode."
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Run with real OpenAI API calls (requires OPENAI_API_KEY). "
             "Overrides --dry-run."
    )
    return parser.parse_args()


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # ── Parse command-line arguments ──────────────────────────────
    args = parse_args()

    # ── Determine run mode ────────────────────────────────────────
    dry_run = not args.live

    # ── Display welcome banner ────────────────────────────────────
    print_banner()

    # ── Show run mode ─────────────────────────────────────────────
    if dry_run:
        print("  ℹ️  Running in DRY-RUN mode (simulated data, no API key needed)")
        print("     To use real LLM calls: python main.py --live --topic \"your topic\"")
    else:
        if not os.environ.get("OPENAI_API_KEY"):
            print("  ❌ Error: OPENAI_API_KEY environment variable not set.")
            print("     Set it with: export OPENAI_API_KEY=your-key-here")
            print("     Or run in dry-run mode: python main.py --dry-run")
            sys.exit(1)
        print("  ℹ️  Running in LIVE mode (real LLM calls)")

    # ── Get topic ─────────────────────────────────────────────────
    topic = args.topic
    if not topic:
        topic = input("\n  Enter research topic: ").strip()
        if not topic:
            topic = "AI agents in enterprise software development"
            print(f"  Using default topic: '{topic}'")

    # ── Run the pipeline ──────────────────────────────────────────
    print(f"\n  🎯 Starting research on: '{topic}'")
    print(f"  ⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    run_pipeline(topic=topic, dry_run=dry_run)

    # ── Done ──────────────────────────────────────────────────────
    print(f"\n  ✅ Pipeline complete at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  📁 Report available at: output/report.md")
    print()
