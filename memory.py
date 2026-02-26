"""
============================================================================
memory.py — Three-Tier Agent Memory System
============================================================================

PURPOSE:
    This module implements a dynamic memory system that agents share during
    workflow execution. It demonstrates how stateful agents maintain context,
    learn facts, and recall past experiences.

ARCHITECTURE:
    The memory has three tiers, inspired by human cognition:

    ┌─────────────────────────────────────────────────────────┐
    │                    AgentMemory                          │
    │                                                        │
    │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
    │  │  SHORT-TERM  │  │  LONG-TERM   │  │   EPISODIC   │  │
    │  │  (working)   │  │  (knowledge) │  │  (experience)│  │
    │  │              │  │              │  │              │  │
    │  │  Current     │  │  Persistent  │  │  Past task   │  │
    │  │  task data,  │  │  facts the   │  │  records     │  │
    │  │  recent      │  │  agent has   │  │  with        │  │
    │  │  observations│  │  learned     │  │  outcomes    │  │
    │  └──────────────┘  └──────────────┘  └──────────────┘  │
    └─────────────────────────────────────────────────────────┘

WHY THREE TIERS?
    - Short-term memory is fast but limited — it holds only what's needed
      RIGHT NOW (like your mental "scratchpad" while solving a problem).
    - Long-term memory persists across tasks — facts learned in one task
      are available in the next (like remembering a colleague's name).
    - Episodic memory records complete task outcomes — the agent can look
      back at what worked and what didn't (like learning from experience).

USAGE IN THIS PROJECT:
    - The Research Agent writes search results to short-term memory.
    - The Analysis Agent reads short-term memory and stores discovered
      trends in long-term memory.
    - The Writer Agent checks episodic memory for past report formats
      that were well-received.
    - All agents share the SAME memory instance via the workflow state.

============================================================================
"""

from datetime import datetime
from collections import deque
from typing import Any
import json


class AgentMemory:
    """
    A shared memory system for multi-agent collaboration.

    This class is the central knowledge store that all agents in the workflow
    can read from and write to. It's injected into the LangGraph state so
    every node (agent) has access to the same memory instance.

    Attributes:
        short_term (deque):  Bounded working memory for the current task.
        long_term (dict):    Unbounded persistent knowledge base.
        episodic (list):     Log of past task executions and their outcomes.
    """

    def __init__(self, max_short_term: int = 100):
        """
        Initialize the three memory tiers.

        Args:
            max_short_term: Maximum number of items in working memory.
                           When full, the oldest item is automatically
                           evicted (FIFO via deque). This prevents memory
                           from growing unbounded during long tasks.
        """
        # ── SHORT-TERM MEMORY ─────────────────────────────────────
        # A bounded queue that holds recent observations, intermediate
        # results, and context for the current task. Using a deque with
        # maxlen ensures automatic eviction of the oldest items when
        # capacity is reached — no manual cleanup needed.
        self.short_term: deque[dict] = deque(maxlen=max_short_term)

        # ── LONG-TERM MEMORY ──────────────────────────────────────
        # A dictionary that stores persistent facts the agent has learned.
        # Unlike short-term memory, this is unbounded and survives across
        # tasks. Each entry tracks when it was learned and how often it
        # has been accessed (useful for measuring knowledge utilization).
        self.long_term: dict[str, dict] = {}

        # ── EPISODIC MEMORY ───────────────────────────────────────
        # A chronological log of past task executions. Each episode
        # records what the task was, what actions were taken, and whether
        # the outcome was successful. This allows the agent to "learn
        # from experience" — e.g., avoiding strategies that failed before.
        self.episodic: list[dict] = []

    # ==================================================================
    # SHORT-TERM MEMORY OPERATIONS
    # ==================================================================
    # These methods manage the agent's "working memory" — the scratchpad
    # for the current task. Items are tagged with timestamps and metadata
    # so agents can filter by recency or source.
    # ==================================================================

    def remember(self, content: str, source: str = "system",
                 metadata: dict = None) -> None:
        """
        Add an observation or fact to short-term working memory.

        This is the most frequently called memory operation. Agents use it
        to store intermediate results (e.g., "Search returned 5 results
        about quantum computing"), observations, or context that other
        agents in the pipeline will need.

        Args:
            content:  The text content to remember.
            source:   Which agent or system wrote this (for traceability).
            metadata: Optional dictionary of extra data (e.g., confidence
                      scores, source URLs, data types).

        Example:
            memory.remember(
                content="Search found 3 articles about AI in healthcare",
                source="research_agent",
                metadata={"query": "AI healthcare", "result_count": 3}
            )
        """
        self.short_term.append({
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
        })

    def get_recent_context(self, n: int = 10,
                           source_filter: str = None) -> list[str]:
        """
        Retrieve the N most recent items from working memory.

        This is how agents "catch up" on what happened before them in
        the pipeline. For example, the Analysis Agent calls this to read
        all the search results the Research Agent stored.

        Args:
            n:             Number of recent items to retrieve.
            source_filter: If set, only return items from this source.
                          Useful when an agent only wants its own notes.

        Returns:
            A list of content strings, most recent last.
        """
        items = list(self.short_term)

        # ── Optional filtering by source ──
        # In a multi-agent system, each agent writes to the same memory.
        # Filtering lets an agent read only items from a specific source
        # (e.g., only research results, not analysis notes).
        if source_filter:
            items = [item for item in items if item["source"] == source_filter]

        # Return only the content strings (not the full metadata dicts)
        # to keep the LLM prompt clean and concise.
        return [item["content"] for item in items[-n:]]

    def get_working_memory_summary(self) -> str:
        """
        Generate a human-readable summary of current working memory.

        This is injected into agent prompts so the LLM knows what context
        is available. Without this, the agent would be "blind" to what
        other agents have done.

        Returns:
            A formatted string summarizing all items in working memory.
        """
        if not self.short_term:
            return "Working memory is empty — no prior context available."

        lines = []
        for i, item in enumerate(self.short_term, 1):
            lines.append(f"  [{i}] ({item['source']}) {item['content']}")
        return f"Working Memory ({len(self.short_term)} items):\n" + "\n".join(lines)

    # ==================================================================
    # LONG-TERM MEMORY OPERATIONS
    # ==================================================================
    # These methods manage persistent knowledge. Unlike short-term memory,
    # facts stored here are never evicted automatically. They represent
    # "things the agent knows" that are reusable across tasks.
    # ==================================================================

    def learn(self, key: str, value: Any, source: str = "system") -> None:
        """
        Store a fact in long-term memory.

        Long-term memory is organized as key-value pairs. Each entry also
        tracks metadata: when it was learned, who wrote it, and how many
        times it has been accessed. This metadata is useful for:
          - Debugging (which agent learned this?)
          - Analytics (which facts are most useful?)
          - Conflict resolution (when was this last updated?)

        Args:
            key:    A descriptive key for the fact (e.g., "top_ai_trend_2024").
            value:  The fact itself (can be any type: str, dict, list, etc.).
            source: Which agent stored this fact.

        Example:
            memory.learn(
                key="top_ai_trend_2024",
                value="Agentic AI systems are the #1 trend",
                source="analysis_agent"
            )
        """
        self.long_term[key] = {
            "value": value,
            "source": source,
            "learned_at": datetime.now().isoformat(),
            "access_count": 0,     # Tracks how often this fact is recalled
            "last_accessed": None,  # Timestamp of most recent access
        }

    def recall(self, key: str) -> Any:
        """
        Retrieve a fact from long-term memory by its key.

        Every recall increments the access counter and updates the
        last_accessed timestamp. This usage tracking helps identify:
          - Frequently used facts (high value — keep them)
          - Never-accessed facts (low value — maybe prune them)

        Args:
            key: The key of the fact to retrieve.

        Returns:
            The stored value, or None if the key doesn't exist.
        """
        if key in self.long_term:
            entry = self.long_term[key]
            # ── Track access patterns ──
            # This is a simple form of "memory reinforcement" — the more
            # a fact is accessed, the more "important" it is considered.
            entry["access_count"] += 1
            entry["last_accessed"] = datetime.now().isoformat()
            return entry["value"]
        return None

    def recall_all(self) -> dict[str, Any]:
        """
        Retrieve ALL facts from long-term memory.

        Used when an agent needs a complete knowledge dump — for example,
        the Writer Agent wants to incorporate all learned facts into the
        final report.

        Returns:
            A dictionary mapping keys to their stored values.
        """
        return {key: entry["value"] for key, entry in self.long_term.items()}

    def get_knowledge_summary(self) -> str:
        """
        Generate a human-readable summary of all long-term knowledge.

        This is injected into agent prompts so they know what facts have
        been learned by previous agents in the pipeline.

        Returns:
            A formatted string listing all known facts.
        """
        if not self.long_term:
            return "Long-term memory is empty — no facts learned yet."

        lines = []
        for key, entry in self.long_term.items():
            lines.append(f"  • {key}: {entry['value']} "
                         f"(by {entry['source']}, "
                         f"accessed {entry['access_count']}x)")
        return f"Known Facts ({len(self.long_term)} items):\n" + "\n".join(lines)

    # ==================================================================
    # EPISODIC MEMORY OPERATIONS
    # ==================================================================
    # These methods manage the agent's "experience log" — a record of
    # complete task executions. This is analogous to how humans remember
    # past events: "Last time I tried X, the result was Y."
    # ==================================================================

    def record_episode(self, task: str, actions: list[str],
                       outcome: str, success: bool,
                       quality_score: float = 0.0) -> None:
        """
        Record a complete task execution as an episode.

        After a workflow completes (successfully or not), this method
        logs the entire experience. Future runs can query this log to
        learn from past successes and failures.

        Args:
            task:          Description of the task that was executed.
            actions:       List of actions/tools used during the task.
            outcome:       Description of the final result.
            success:       Whether the task was completed successfully.
            quality_score: A 1-10 quality rating (from the reviewer agent).

        Example:
            memory.record_episode(
                task="Research AI trends for Q4 report",
                actions=["search_web", "query_database", "analyze_trends"],
                outcome="Generated 2-page report with 5 key findings",
                success=True,
                quality_score=8.5
            )
        """
        self.episodic.append({
            "task": task,
            "actions": actions,
            "outcome": outcome,
            "success": success,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
        })

    def find_similar_episodes(self, task_description: str) -> list[dict]:
        """
        Search episodic memory for tasks similar to the given description.

        This is a simple keyword-based similarity search. In a production
        system, you would use embedding-based similarity (e.g., cosine
        similarity on sentence embeddings) for much better results.

        Args:
            task_description: Description of the current task.

        Returns:
            A list of past episodes that share keywords with the task.
        """
        # ── Simple keyword matching ──
        # Split the task description into individual words and check
        # if any of them appear in past episode task descriptions.
        # This is a "good enough" approach for demonstration — production
        # systems should use vector similarity search (e.g., FAISS, Pinecone).
        keywords = set(task_description.lower().split())
        similar = []
        for episode in self.episodic:
            episode_keywords = set(episode["task"].lower().split())
            # If at least 2 keywords overlap, consider it "similar"
            if len(keywords & episode_keywords) >= 2:
                similar.append(episode)
        return similar

    def get_success_rate(self) -> float:
        """
        Calculate the overall success rate from episodic memory.

        This metric helps monitor agent performance over time. A dropping
        success rate might indicate prompt degradation or tool failures.

        Returns:
            A float between 0.0 and 1.0 representing the success ratio.
        """
        if not self.episodic:
            return 0.0
        successes = sum(1 for ep in self.episodic if ep["success"])
        return successes / len(self.episodic)

    def get_average_quality(self) -> float:
        """
        Calculate the average quality score across all episodes.

        Returns:
            Average quality score (0.0-10.0), or 0.0 if no episodes exist.
        """
        scored = [ep for ep in self.episodic if ep["quality_score"] > 0]
        if not scored:
            return 0.0
        return sum(ep["quality_score"] for ep in scored) / len(scored)

    def get_episodes_summary(self) -> str:
        """
        Generate a human-readable summary of past episodes.

        This is injected into agent prompts so they can learn from past
        executions — e.g., "Last time we researched this topic, we got
        a quality score of 8/10 using strategy X."

        Returns:
            A formatted string summarizing all past episodes.
        """
        if not self.episodic:
            return "No past episodes recorded yet — this is the first run."

        lines = []
        for i, ep in enumerate(self.episodic, 1):
            status = "✅" if ep["success"] else "❌"
            lines.append(
                f"  {status} Episode {i}: {ep['task']} → "
                f"Quality: {ep['quality_score']}/10 | "
                f"Actions: {', '.join(ep['actions'])}"
            )

        rate = self.get_success_rate()
        avg_q = self.get_average_quality()
        header = (f"Past Episodes ({len(self.episodic)} total, "
                  f"{rate:.0%} success rate, avg quality: {avg_q:.1f}/10):")
        return header + "\n" + "\n".join(lines)

    # ==================================================================
    # SERIALIZATION
    # ==================================================================
    # These methods convert the memory to/from a dictionary so it can be
    # stored in LangGraph's state (which must be JSON-serializable).
    # ==================================================================

    def to_dict(self) -> dict:
        """
        Serialize the entire memory to a JSON-compatible dictionary.

        LangGraph state must be serializable (for checkpointing and
        persistence). This method converts all three memory tiers into
        a plain dictionary that can be stored in the state.

        Returns:
            A dictionary containing all memory data.
        """
        return {
            "short_term": list(self.short_term),
            "long_term": self.long_term,
            "episodic": self.episodic,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentMemory":
        """
        Reconstruct an AgentMemory instance from a serialized dictionary.

        This is the inverse of to_dict(). It's used when loading state
        from a LangGraph checkpoint — the memory data is stored as a
        dict in the state, and this method reconstructs the full object.

        Args:
            data: A dictionary previously created by to_dict().

        Returns:
            A fully populated AgentMemory instance.
        """
        memory = cls()
        for item in data.get("short_term", []):
            memory.short_term.append(item)
        memory.long_term = data.get("long_term", {})
        memory.episodic = data.get("episodic", [])
        return memory

    def __repr__(self) -> str:
        """Concise string representation for debugging."""
        return (
            f"AgentMemory("
            f"short_term={len(self.short_term)} items, "
            f"long_term={len(self.long_term)} facts, "
            f"episodic={len(self.episodic)} episodes)"
        )
