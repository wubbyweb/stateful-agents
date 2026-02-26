# Multi-Agent Research System — How to Run & What to Observe

## Prerequisites

- **Python 3.10+** installed
- **uv** package manager (or pip)
- No API key needed — the project runs in dry-run mode with simulated data

---

## Setup (One Time)

```bash
cd ~/Programs/stateful-agents

# Create a virtual environment and install dependencies
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

---

## Running the Pipeline

```bash
# Activate the environment (if not already active)
source .venv/bin/activate

# Run with a topic of your choice
python main.py --topic "AI agents in healthcare" --dry-run

# Or let it prompt you for a topic
python main.py --dry-run

# Try different topics to see how results change
python main.py --topic "Future of autonomous vehicles" --dry-run
python main.py --topic "Climate change mitigation strategies" --dry-run
```

---

## What to Observe

### 1. Agent Execution Order

Watch the four agents execute in sequence:

| Step | Agent | What It Does |
|------|-------|-------------|
| 1 | 🔍 **Research** | Searches the web and queries a database for raw data |
| 2 | 📊 **Analysis** | Extracts trends, performs SWOT analysis, calculates metrics |
| 3 | ✍️ **Writer** | Generates a polished markdown report from the analysis |
| 4 | ⭐ **Reviewer** | Scores the report (1–10) and decides: approve or revise |

### 2. The Feedback Loop

The quality reviewer scores the report. If the score is **below 7/10**, the entire pipeline loops back to the Research Agent for another pass. Watch the console for:

```
⭐  Agent: REVIEWER
    ↳ Quality: 8.5/10 — ✅ APPROVED (iteration 1)
```

If you modify `_evaluate_report_quality()` in `agents.py` to return lower scores, you'll see the loop-back behavior in action.

### 3. Memory Tracking

Each agent's output shows the memory state. Notice how it grows:

```
Research:  7 short-term,  3 long-term, 0 episodic
Analysis:  9 short-term,  6 long-term, 0 episodic   ← learned 3 new facts
Writer:   11 short-term,  6 long-term, 0 episodic
Reviewer: 12 short-term,  6 long-term, 1 episodic   ← recorded the episode
```

- **Short-term** grows as each agent logs its activity
- **Long-term** jumps when the Analysis Agent stores discovered facts
- **Episodic** gets its first entry when the Reviewer records the task outcome

### 4. The Action Log

At the end, you'll see every action the agents took:

```
 1. search_web('AI agents in healthcare')
 2. search_web('AI agents in healthcare trends statistics')
 3. search_web('AI agents in healthcare challenges risks')
 4. query_database('business metrics overview')
 5. query_database('active customer data')
 6. calculate('2500000 / 4')
 7. calculate('((3100000 - 1200000) / 1200000) * 100')
 8. write_report
 9. quality_review(score=8.5)
```

This gives you full traceability of what the system did and why.

### 5. The Generated Report

The final report is saved to **`output/report.md`**. Open it and observe:

- **Executive Summary** with calculated metrics (revenue, growth %)
- **Research Methodology** table showing which agent did what
- **SWOT Analysis** derived from the simulated data
- **Recommendations** based on the analysis
- **Memory State Appendix** showing what the agents remembered

---

## Project Structure

```
stateful-agents/
├── main.py          ← Run this (entry point)
├── workflow.py      ← LangGraph graph with conditional edges
├── agents.py        ← 4 agent functions + quality scoring
├── state.py         ← Shared state definition (TypedDict)
├── memory.py        ← 3-tier memory system
├── tools.py         ← Simulated search/db/calc tools
├── requirements.txt ← Dependencies
├── .env.example     ← API key template (not needed for dry-run)
└── output/
    └── report.md    ← Generated after running the pipeline
```

---

## Key Learning Points

1. **Read the code comments** — every file is heavily annotated explaining *why*, not just *what*
2. **Agents communicate through shared state** — look at how `state["search_results"]` flows from Research → Analysis
3. **Memory enables learning** — the episodic record from one run can influence the next
4. **The conditional edge is what makes this a graph** — without the reviewer loop-back, this would just be a simple chain
5. **Dry-run mode proves the architecture works** — no LLM needed to validate the orchestration logic
