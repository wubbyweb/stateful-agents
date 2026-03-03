"""
============================================================================
memory.py — Distributed Three-Tier Agent Memory System
============================================================================

PURPOSE:
    This module implements a production-grade memory system for multi-agent
    workflows. It supports three backends — Azure Cache for Redis (caching),
    Azure Cosmos DB via the MongoDB API (durable persistence), and a local
    in-process fallback — wired together behind the same AgentMemory API
    that the rest of the codebase already uses.

WHY DISTRIBUTED?
    Local in-memory state (deque / dict / list) is fine for a single-process
    demo, but in production you need:
      • Persistence across process restarts
      • Sub-millisecond caching for "hot" working memory (Redis)
      • Durable, indexed storage for long-term facts & episodic logs (Cosmos)
      • Concurrent agent isolation via unique thread/execution IDs

ARCHITECTURE:
    ┌──────────────────────────────────────────────────────────────────┐
    │                        AgentMemory                              │
    │                   (thread_id scoped)                            │
    │                                                                  │
    │  ┌──────────────────┐  ┌──────────────────┐  ┌───────────────┐  │
    │  │ SHORT-TERM       │  │ LONG-TERM        │  │ EPISODIC      │  │
    │  │ (Redis cache)    │  │ (Cosmos MongoDB) │  │ (Cosmos)      │  │
    │  │                  │  │                  │  │               │  │
    │  │ Current working  │  │ Persistent facts │  │ Task history  │  │
    │  │ memory, TTL-     │  │ indexed by key   │  │ with outcomes │  │
    │  │ managed eviction │  │ + thread_id      │  │ + scores      │  │
    │  └──────────────────┘  └──────────────────┘  └───────────────┘  │
    │                                                                  │
    │  ┌──────────────────────────────────────────────────────────────┐│
    │  │              LOCAL FALLBACK (in-process)                     ││
    │  │  Activates automatically when Redis / Cosmos are unreachable ││
    │  └──────────────────────────────────────────────────────────────┘│
    └──────────────────────────────────────────────────────────────────┘

THREAD ID ISOLATION:
    Every memory key is prefixed with a `thread_id`.  Different
    conversations / executions / tasks get their own isolated namespace
    so concurrent pipelines never collide.

ENVIRONMENT VARIABLES (see .env.example):
    AZURE_COSMOS_CONNECTION_STRING   – MongoDB-compatible connection string
    AZURE_COSMOS_DATABASE            – database name (default: agent_memory)
    AZURE_REDIS_HOST                 – hostname for Azure Cache for Redis
    AZURE_REDIS_PORT                 – port (default: 6380)
    AZURE_REDIS_PASSWORD             – access key
    AZURE_REDIS_SSL                  – "true" for TLS (default: true)

============================================================================
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from collections import deque
from datetime import datetime
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── Optional imports — graceful degradation if libraries aren't installed ──
try:
    import redis
    _REDIS_AVAILABLE = True
except ImportError:
    _REDIS_AVAILABLE = False
    logger.info("redis-py not installed — Redis backend unavailable.")

try:
    from pymongo import MongoClient
    from pymongo.errors import ConnectionFailure
    _COSMOS_AVAILABLE = True
except ImportError:
    _COSMOS_AVAILABLE = False
    logger.info("pymongo not installed — Cosmos DB backend unavailable.")


# ============================================================================
# BACKEND HELPERS
# ============================================================================

class _RedisBackend:
    """
    Thin wrapper around Azure Cache for Redis.

    All keys are namespaced by thread_id to ensure concurrent pipelines
    never overlap.  Short-term memory items are stored as a Redis List
    (bounded via LTRIM), while long-term and episodic data use Redis
    only as a write-through cache with configurable TTL.
    """

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self._client: Optional["redis.Redis"] = None
        self._connected = False

        host = os.getenv("AZURE_REDIS_HOST", "")
        port = int(os.getenv("AZURE_REDIS_PORT", "6380"))
        password = os.getenv("AZURE_REDIS_PASSWORD", "")
        use_ssl = os.getenv("AZURE_REDIS_SSL", "true").lower() == "true"

        if not host or not _REDIS_AVAILABLE:
            logger.info("Redis backend: no host configured or redis-py "
                        "missing — skipping.")
            return

        try:
            self._client = redis.Redis(
                host=host,
                port=port,
                password=password,
                ssl=use_ssl,
                decode_responses=True,
                socket_connect_timeout=5,
                retry_on_timeout=True,
            )
            self._client.ping()
            self._connected = True
            logger.info("Redis backend: connected to %s:%s", host, port)
        except Exception as exc:
            logger.warning("Redis backend: could not connect — %s", exc)
            self._client = None

    # ── key helpers ────────────────────────────────────────────────
    def _key(self, namespace: str) -> str:
        return f"agent_memory:{self.thread_id}:{namespace}"

    @property
    def connected(self) -> bool:
        return self._connected and self._client is not None

    # ── Short-term (list) ─────────────────────────────────────────
    def push_short_term(self, item: dict, max_len: int = 100) -> None:
        if not self.connected:
            return
        key = self._key("short_term")
        self._client.rpush(key, json.dumps(item))
        self._client.ltrim(key, -max_len, -1)

    def get_short_term(self) -> list[dict]:
        if not self.connected:
            return []
        raw = self._client.lrange(self._key("short_term"), 0, -1)
        return [json.loads(r) for r in raw]

    # ── Long-term (hash) ──────────────────────────────────────────
    def set_long_term(self, key: str, entry: dict) -> None:
        if not self.connected:
            return
        self._client.hset(self._key("long_term"), key, json.dumps(entry))

    def get_long_term(self, key: str) -> Optional[dict]:
        if not self.connected:
            return None
        raw = self._client.hget(self._key("long_term"), key)
        return json.loads(raw) if raw else None

    def get_all_long_term(self) -> dict[str, dict]:
        if not self.connected:
            return {}
        raw_all = self._client.hgetall(self._key("long_term"))
        return {k: json.loads(v) for k, v in raw_all.items()}

    # ── Episodic (list) ───────────────────────────────────────────
    def push_episodic(self, episode: dict) -> None:
        if not self.connected:
            return
        self._client.rpush(self._key("episodic"), json.dumps(episode))

    def get_all_episodic(self) -> list[dict]:
        if not self.connected:
            return []
        raw = self._client.lrange(self._key("episodic"), 0, -1)
        return [json.loads(r) for r in raw]


class _CosmosBackend:
    """
    Thin wrapper around Azure Cosmos DB (MongoDB API).

    Three collections mirror the three memory tiers:
      • short_term  – working observations (secondary durability)
      • long_term   – persistent facts
      • episodic    – task execution history

    Every document contains a `thread_id` field that is indexed for
    fast per-conversation lookups.
    """

    def __init__(self, thread_id: str):
        self.thread_id = thread_id
        self._db = None
        self._connected = False

        conn_str = os.getenv("AZURE_COSMOS_CONNECTION_STRING", "")
        db_name = os.getenv("AZURE_COSMOS_DATABASE", "agent_memory")

        if not conn_str or not _COSMOS_AVAILABLE:
            logger.info("Cosmos DB backend: no connection string or "
                        "pymongo missing — skipping.")
            return

        try:
            client = MongoClient(
                conn_str,
                serverSelectionTimeoutMS=5000,
                retryWrites=False,      # Cosmos DB MongoDB API constraint
            )
            # Verify connectivity
            client.admin.command("ping")
            self._db = client[db_name]

            # Ensure indexes on thread_id for every collection
            for coll_name in ("short_term", "long_term", "episodic"):
                self._db[coll_name].create_index("thread_id")

            self._connected = True
            logger.info("Cosmos DB backend: connected to database '%s'",
                        db_name)
        except Exception as exc:
            logger.warning("Cosmos DB backend: could not connect — %s", exc)
            self._db = None

    @property
    def connected(self) -> bool:
        return self._connected and self._db is not None

    # ── Short-term ────────────────────────────────────────────────
    def insert_short_term(self, item: dict) -> None:
        if not self.connected:
            return
        doc = {**item, "thread_id": self.thread_id}
        self._db["short_term"].insert_one(doc)

    def get_short_term(self, limit: int = 100) -> list[dict]:
        if not self.connected:
            return []
        cursor = (
            self._db["short_term"]
            .find({"thread_id": self.thread_id})
            .sort("timestamp", 1)
            .limit(limit)
        )
        return [{k: v for k, v in doc.items() if k != "_id"}
                for doc in cursor]

    # ── Long-term ─────────────────────────────────────────────────
    def upsert_long_term(self, key: str, entry: dict) -> None:
        if not self.connected:
            return
        self._db["long_term"].update_one(
            {"thread_id": self.thread_id, "key": key},
            {"$set": {**entry, "thread_id": self.thread_id, "key": key}},
            upsert=True,
        )

    def get_long_term(self, key: str) -> Optional[dict]:
        if not self.connected:
            return None
        doc = self._db["long_term"].find_one(
            {"thread_id": self.thread_id, "key": key}
        )
        if doc:
            return {k: v for k, v in doc.items()
                    if k not in ("_id", "thread_id", "key")}
        return None

    def get_all_long_term(self) -> dict[str, dict]:
        if not self.connected:
            return {}
        cursor = self._db["long_term"].find(
            {"thread_id": self.thread_id}
        )
        result = {}
        for doc in cursor:
            key = doc.get("key", "")
            result[key] = {k: v for k, v in doc.items()
                           if k not in ("_id", "thread_id", "key")}
        return result

    # ── Episodic ──────────────────────────────────────────────────
    def insert_episodic(self, episode: dict) -> None:
        if not self.connected:
            return
        doc = {**episode, "thread_id": self.thread_id}
        self._db["episodic"].insert_one(doc)

    def get_all_episodic(self) -> list[dict]:
        if not self.connected:
            return []
        cursor = (
            self._db["episodic"]
            .find({"thread_id": self.thread_id})
            .sort("timestamp", 1)
        )
        return [{k: v for k, v in doc.items() if k != "_id"}
                for doc in cursor]


# ============================================================================
# MAIN CLASS
# ============================================================================

class AgentMemory:
    """
    A distributed, thread-isolated memory system for multi-agent collaboration.

    Backends:
        1. Redis  – ultra-low-latency cache for short-term / hot data
        2. Cosmos – durable persistence for long-term facts & episodes
        3. Local  – automatic in-process fallback if cloud services are down

    Every public method writes to ALL available backends (write-through).
    Reads prefer Redis → Cosmos → local, falling back transparently.

    Attributes:
        thread_id  (str):   Unique execution / conversation identifier.
        short_term (deque): Bounded working memory for the current task.
        long_term  (dict):  Unbounded persistent knowledge base.
        episodic   (list):  Log of past task executions and their outcomes.
    """

    def __init__(
        self,
        max_short_term: int = 100,
        thread_id: str | None = None,
    ):
        """
        Initialize the three memory tiers and connect to backends.

        Args:
            max_short_term: Maximum number of items in working memory.
            thread_id:      Unique identifier for this execution context.
                            If not provided, a new UUID is generated so
                            that every run is isolated by default.
        """
        # ── Execution isolation ───────────────────────────────────
        self.thread_id: str = thread_id or str(uuid.uuid4())

        # ── LOCAL TIERS (always available) ────────────────────────
        self.short_term: deque[dict] = deque(maxlen=max_short_term)
        self.long_term: dict[str, dict] = {}
        self.episodic: list[dict] = []
        self._max_short_term = max_short_term

        # ── DISTRIBUTED BACKENDS ──────────────────────────────────
        self._redis = _RedisBackend(self.thread_id)
        self._cosmos = _CosmosBackend(self.thread_id)

        # On init, hydrate local state FROM distributed stores so the
        # in-process view is always up-to-date.
        self._hydrate_from_backends()

    # ────────────────────────────────────────────────────────────────
    # HYDRATION — populate local state from distributed stores
    # ────────────────────────────────────────────────────────────────

    def _hydrate_from_backends(self) -> None:
        """
        Pull any existing data for this thread_id from Redis / Cosmos
        into the local in-process structures.  Precedence:
          Redis (fastest) → Cosmos (durable) → leave empty.
        """
        # -- Short-term --
        items = self._redis.get_short_term()
        if not items:
            items = self._cosmos.get_short_term(self._max_short_term)
        for item in items:
            self.short_term.append(item)

        # -- Long-term --
        lt = self._redis.get_all_long_term()
        if not lt:
            lt = self._cosmos.get_all_long_term()
        self.long_term.update(lt)

        # -- Episodic --
        eps = self._redis.get_all_episodic()
        if not eps:
            eps = self._cosmos.get_all_episodic()
        self.episodic.extend(eps)

    # ==================================================================
    # SHORT-TERM MEMORY OPERATIONS
    # ==================================================================

    def remember(self, content: str, source: str = "system",
                 metadata: dict = None) -> None:
        """
        Add an observation or fact to short-term working memory.

        Writes to ALL available backends (local + Redis + Cosmos).

        Args:
            content:  The text content to remember.
            source:   Which agent or system wrote this.
            metadata: Optional extra data (confidence scores, URLs, etc.).
        """
        item = {
            "content": content,
            "source": source,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {},
            "thread_id": self.thread_id,
        }

        # ── Local ──
        self.short_term.append(item)

        # ── Redis (fast cache) ──
        self._redis.push_short_term(item, self._max_short_term)

        # ── Cosmos (durable) ──
        self._cosmos.insert_short_term(item)

    def get_recent_context(self, n: int = 10,
                           source_filter: str = None) -> list[str]:
        """
        Retrieve the N most recent items from working memory.

        Args:
            n:             Number of recent items to retrieve.
            source_filter: If set, only return items from this source.

        Returns:
            A list of content strings, most recent last.
        """
        items = list(self.short_term)
        if source_filter:
            items = [i for i in items if i["source"] == source_filter]
        return [item["content"] for item in items[-n:]]

    def get_working_memory_summary(self) -> str:
        """Generate a human-readable summary of current working memory."""
        if not self.short_term:
            return "Working memory is empty — no prior context available."

        lines = []
        for i, item in enumerate(self.short_term, 1):
            lines.append(f"  [{i}] ({item['source']}) {item['content']}")
        return (f"Working Memory ({len(self.short_term)} items, "
                f"thread={self.thread_id[:8]}…):\n" + "\n".join(lines))

    # ==================================================================
    # LONG-TERM MEMORY OPERATIONS
    # ==================================================================

    def learn(self, key: str, value: Any, source: str = "system") -> None:
        """
        Store a fact in long-term memory.

        Writes through to Redis (cache) AND Cosmos (durable).

        Args:
            key:    A descriptive key for the fact.
            value:  The fact itself (any JSON-serialisable type).
            source: Which agent stored this fact.
        """
        entry = {
            "value": value,
            "source": source,
            "learned_at": datetime.now().isoformat(),
            "access_count": 0,
            "last_accessed": None,
            "thread_id": self.thread_id,
        }

        # ── Local ──
        self.long_term[key] = entry

        # ── Redis cache ──
        self._redis.set_long_term(key, entry)

        # ── Cosmos durable ──
        self._cosmos.upsert_long_term(key, entry)

    def recall(self, key: str) -> Any:
        """
        Retrieve a fact from long-term memory by its key.

        Increments the access counter and updates last_accessed.
        Writes the updated entry back to all backends.

        Args:
            key: The key of the fact to retrieve.

        Returns:
            The stored value, or None if the key doesn't exist.
        """
        if key in self.long_term:
            entry = self.long_term[key]
            entry["access_count"] += 1
            entry["last_accessed"] = datetime.now().isoformat()

            # Propagate usage stats to distributed stores
            self._redis.set_long_term(key, entry)
            self._cosmos.upsert_long_term(key, entry)

            return entry["value"]
        return None

    def recall_all(self) -> dict[str, Any]:
        """
        Retrieve ALL facts from long-term memory.

        Returns:
            A dictionary mapping keys to their stored values.
        """
        return {key: entry["value"]
                for key, entry in self.long_term.items()}

    def get_knowledge_summary(self) -> str:
        """Generate a human-readable summary of all long-term knowledge."""
        if not self.long_term:
            return "Long-term memory is empty — no facts learned yet."

        lines = []
        for key, entry in self.long_term.items():
            lines.append(
                f"  • {key}: {entry['value']} "
                f"(by {entry['source']}, "
                f"accessed {entry['access_count']}x)"
            )
        return (f"Known Facts ({len(self.long_term)} items, "
                f"thread={self.thread_id[:8]}…):\n" + "\n".join(lines))

    # ==================================================================
    # EPISODIC MEMORY OPERATIONS
    # ==================================================================

    def record_episode(self, task: str, actions: list[str],
                       outcome: str, success: bool,
                       quality_score: float = 0.0) -> None:
        """
        Record a complete task execution as an episode.

        Writes through to Redis AND Cosmos.

        Args:
            task:          Description of the task that was executed.
            actions:       List of actions/tools used during the task.
            outcome:       Description of the final result.
            success:       Whether the task was completed successfully.
            quality_score: A 1-10 quality rating.
        """
        episode = {
            "task": task,
            "actions": actions,
            "outcome": outcome,
            "success": success,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
            "thread_id": self.thread_id,
        }

        # ── Local ──
        self.episodic.append(episode)

        # ── Redis ──
        self._redis.push_episodic(episode)

        # ── Cosmos ──
        self._cosmos.insert_episodic(episode)

    def find_similar_episodes(self, task_description: str) -> list[dict]:
        """
        Search episodic memory for tasks similar to the given description.

        Uses simple keyword overlap.  In production, replace with
        embedding-based similarity search (e.g., Cosmos vector search).

        Args:
            task_description: Description of the current task.

        Returns:
            Past episodes that share ≥ 2 keywords with the description.
        """
        keywords = set(task_description.lower().split())
        similar = []
        for episode in self.episodic:
            episode_keywords = set(episode["task"].lower().split())
            if len(keywords & episode_keywords) >= 2:
                similar.append(episode)
        return similar

    def get_success_rate(self) -> float:
        """
        Calculate the overall success rate from episodic memory.

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
            Average quality score (0.0-10.0), or 0.0 if none.
        """
        scored = [ep for ep in self.episodic if ep["quality_score"] > 0]
        if not scored:
            return 0.0
        return sum(ep["quality_score"] for ep in scored) / len(scored)

    def get_episodes_summary(self) -> str:
        """Generate a human-readable summary of past episodes."""
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
                  f"{rate:.0%} success rate, avg quality: {avg_q:.1f}/10, "
                  f"thread={self.thread_id[:8]}…):")
        return header + "\n" + "\n".join(lines)

    # ==================================================================
    # SERIALIZATION  (LangGraph state ↔ AgentMemory)
    # ==================================================================

    def to_dict(self) -> dict:
        """
        Serialize the entire memory to a JSON-compatible dictionary.

        Includes the thread_id so it can be round-tripped through
        LangGraph state checkpointing.

        Returns:
            A dictionary containing all memory data + thread_id.
        """
        return {
            "thread_id": self.thread_id,
            "short_term": list(self.short_term),
            "long_term": self.long_term,
            "episodic": self.episodic,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentMemory":
        """
        Reconstruct an AgentMemory instance from a serialized dictionary.

        If a thread_id is present in *data* the same ID is reused so
        distributed backends resume from the correct namespace.

        Args:
            data: A dictionary previously created by to_dict().

        Returns:
            A fully populated AgentMemory instance.
        """
        thread_id = data.get("thread_id")
        memory = cls(thread_id=thread_id)

        # ── Overlay serialized data ──
        # Clear whatever was hydrated from backends, then load from dict.
        memory.short_term.clear()
        for item in data.get("short_term", []):
            memory.short_term.append(item)
        memory.long_term = data.get("long_term", {})
        memory.episodic = data.get("episodic", [])
        return memory

    # ==================================================================
    # CONVENIENCE / DIAGNOSTICS
    # ==================================================================

    @property
    def backend_status(self) -> dict[str, bool]:
        """Return connectivity status for each backend."""
        return {
            "local": True,
            "redis": self._redis.connected,
            "cosmos_db": self._cosmos.connected,
        }

    def __repr__(self) -> str:
        """Concise string representation for debugging."""
        backends = ", ".join(
            name for name, ok in self.backend_status.items() if ok
        )
        return (
            f"AgentMemory("
            f"thread={self.thread_id[:8]}…, "
            f"short_term={len(self.short_term)} items, "
            f"long_term={len(self.long_term)} facts, "
            f"episodic={len(self.episodic)} episodes, "
            f"backends=[{backends}])"
        )
