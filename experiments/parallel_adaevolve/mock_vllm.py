"""
Mock vLLM that emulates prefix caching with the same hashing semantics as
the real engine, plus a bandwidth-limited prefill stage so concurrent
calls actually contend for compute. Used by ``bench.py`` to compare two
arms (K independent AdaEvolve runs vs 1 run × K-batched candidates) on
identical workloads.

Caching model (matches vLLM v0.18 description in KV_CACHING_REPORT.md):
  * 16-token blocks.
  * Chained SHA-256: hash_i = SHA256(hash_{i-1} || tokens_i)
  * Lookup walks the chain; first miss invalidates everything after.
  * LRU eviction beyond max_blocks.
  * Concurrent COMPUTING coordination: when request B asks for a block
    that request A is already computing, B awaits A's future and sees a
    cache HIT once A finishes. This is what gives K-batched same-prompt
    its near-zero marginal prefill cost.

Latency model:
  * UNCACHED_BLOCK_US = block_size * 60us  (≈ vLLM B200 prefill)
  * CACHED_BLOCK_US   = block_size *  3us  (memory read)
  * DECODE_US_PER_TOK = 10_000us           (10 ms / token)
  * The GPU has a global ``prefill_lock`` so only one request's prefill
    work runs at a time. Decode is treated as effectively parallel via
    continuous batching (no lock).

This is a simulation, not byte-exact vLLM; the goal is to demonstrate
the cache-sharing properties accurately enough to inform algorithmic
decisions, not to reproduce vLLM kernel timings.
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

BLOCK_SIZE = 16
UNCACHED_BLOCK_US = BLOCK_SIZE * 60   # 960us per uncached 16-tok block
CACHED_BLOCK_US = BLOCK_SIZE * 3      # 48us per cached 16-tok block
# When a block is reused via diff-aware (content-only) matching, vLLM still
# has to re-apply RoPE for the new positions. We model that as ~6x faster
# than uncached prefill but ~3x slower than a regular cache hit. This is
# the cost of a "Tier 3.1 diff-aware reuse" hit.
DIFFAWARE_REUSE_BLOCK_US = BLOCK_SIZE * 10   # 160us per diff-aware reused block
DECODE_US_PER_TOK = 10_000            # 10 ms per output token

_TOKEN_RE = re.compile(r"\S+|\s+")


def fake_tokenize(text: str) -> List[str]:
    """Deterministic tokenization — every non-space run + every space run is one token.

    Not a real BPE tokenizer; just gives stable, reproducible token counts
    that scale with prompt length the way a real tokenizer would.
    """
    return _TOKEN_RE.findall(text)


def chain_hash_blocks(tokens: List[str], block_size: int = BLOCK_SIZE) -> List[str]:
    """Slice tokens into block_size chunks and chain-hash each block."""
    hashes: List[str] = []
    prev = b"\x00" * 32
    for i in range(0, len(tokens), block_size):
        block = tokens[i : i + block_size]
        if len(block) < block_size:
            # Real vLLM only caches *complete* blocks; tail is recomputed.
            # We mark the tail as uncacheable by giving it a unique hash.
            tail_id = hashlib.sha256(("__tail__" + str(time.time_ns())).encode()).digest()
            hashes.append(tail_id.hex())
            break
        h = hashlib.sha256(prev + ("\x1f".join(block)).encode("utf-8")).digest()
        hashes.append(h.hex())
        prev = h
    return hashes


def content_hash_blocks(tokens: List[str], block_size: int = BLOCK_SIZE) -> List[str]:
    """Position-independent content hash of each block. Used for diff-aware
    KV reuse: identical block content can be matched by hash regardless of
    where in the sequence it sits, at the cost of one RoPE re-application
    per reused block."""
    hashes: List[str] = []
    for i in range(0, len(tokens), block_size):
        block = tokens[i : i + block_size]
        if len(block) < block_size:
            hashes.append("")  # tail — never cached
            break
        h = hashlib.sha256(("\x1f".join(block)).encode("utf-8")).digest()
        hashes.append(h.hex())
    return hashes


@dataclass
class _BlockEntry:
    state: str  # "COMPUTING" | "READY"
    ready_event: asyncio.Event
    last_used_ts: float
    priority: int = 0    # higher = pinned harder against eviction
    tag: str = ""        # "system" / "parent_island_<i>" / "sibling_iter_<t>" / "user"


@dataclass
class CallStats:
    request_id: str
    arrived_at: float
    finished_at: float
    prompt_tokens: int
    output_tokens: int
    total_blocks: int
    cached_blocks: int          # blocks served from cache (hits)
    computed_blocks: int        # blocks this request had to compute itself
    awaited_blocks: int         # blocks where another request was COMPUTING — "shared hit"
    prefill_us: float
    decode_us: float
    queued_us: float            # time waiting for prefill_lock
    prompt_signature: str       # short hash for grouping in analysis


@dataclass
class GlobalStats:
    calls: List[CallStats] = field(default_factory=list)
    total_prompt_tokens: int = 0
    total_output_tokens: int = 0
    total_blocks_seen: int = 0
    cache_hits: int = 0           # READY at lookup time (chained match)
    shared_hits: int = 0          # COMPUTING at lookup time, awaited
    misses: int = 0               # had to compute fresh
    diff_aware_hits: int = 0      # Tier 3.1: reused via content hash + RoPE
    evictions: int = 0
    peak_blocks: int = 0


class MockVLLM:
    """Process-internal mock vLLM-like engine."""

    def __init__(
        self,
        max_blocks: int = 50_000,
        prefill_batch_size: int = 16,
        decode_concurrency: int = 64,
        seed: int = 42,
        eviction_policy: str = "lru",   # "lru" | "priority"
        enable_diff_aware: bool = False,
    ):
        self.max_blocks = max_blocks
        self._cache: Dict[str, _BlockEntry] = {}
        # vLLM continuous batching: up to `prefill_batch_size` requests can be
        # in their prefill phase at once. Concurrent requests for blocks
        # already marked COMPUTING by a peer await its event (no GPU cost).
        self._prefill_sem = asyncio.Semaphore(prefill_batch_size)
        self._decode_sem = asyncio.Semaphore(decode_concurrency)
        self.stats = GlobalStats()
        self._rng = random.Random(seed)
        self._req_counter = 0
        self._t0 = time.monotonic()
        # Eviction policy. "lru" = vanilla vLLM. "priority" = AdaEvolve-aware
        # (system blocks last, sibling first, parent islands by recency).
        self.eviction_policy = eviction_policy
        # Tier 3.1 — diff-aware cross-iteration KV reuse. When enabled, also
        # maintain a content-only hash index so that blocks whose content
        # matches a previously-seen block can be reused even if their
        # position in the sequence has shifted (with a small RoPE cost).
        self.enable_diff_aware = enable_diff_aware
        self._content_index: Dict[str, str] = {}  # content_hash -> chained_hash

    # ------------------------------------------------------------------
    # Cache mechanics
    # ------------------------------------------------------------------

    def _evict_if_needed(self) -> None:
        excess = len(self._cache) - self.max_blocks
        if excess <= 0:
            return
        ready_items = [
            (h, e) for h, e in self._cache.items() if e.state == "READY"
        ]
        if self.eviction_policy == "priority":
            # AdaEvolve-aware: lower priority first (sibling=0, user=1,
            # parent_island=2, system=3), break ties by oldest last_used_ts.
            ready_items.sort(key=lambda x: (x[1].priority, x[1].last_used_ts))
        else:
            ready_items.sort(key=lambda x: x[1].last_used_ts)
        for h, _ in ready_items[:excess]:
            self._cache.pop(h, None)
            self.stats.evictions += 1

    # ------------------------------------------------------------------
    # Persistent KV — Tier 2.3
    # ------------------------------------------------------------------

    def export_warm_blocks(self, min_priority: int = 3) -> Dict[str, Dict]:
        """Snapshot blocks at or above ``min_priority`` (default: system only).

        Returned mapping is a JSON-friendly representation of the block
        cache; the next ``MockVLLM`` instance can load it via
        ``import_warm_blocks`` to skip the cold-start prefill of those
        blocks. Models the persistent-KV-on-disk feature.
        """
        return {
            h: {
                "priority": e.priority,
                "tag": e.tag,
                "last_used_ts": e.last_used_ts,
            }
            for h, e in self._cache.items()
            if e.state == "READY" and e.priority >= min_priority
        }

    def import_warm_blocks(self, snapshot: Dict[str, Dict]) -> int:
        """Mark blocks from a previous snapshot as ``READY`` in this cache.

        Returns the number of blocks installed. Subsequent requests whose
        chained-hashes match these will hit the cache without paying the
        cold prefill — exactly what a persistent on-disk KV cache buys.
        """
        n = 0
        ev = asyncio.Event()
        ev.set()
        now = time.monotonic()
        for h, meta in snapshot.items():
            if h in self._cache:
                continue
            self._cache[h] = _BlockEntry(
                state="READY",
                ready_event=ev,
                last_used_ts=now,
                priority=int(meta.get("priority", 0)),
                tag=str(meta.get("tag", "")),
            )
            n += 1
        return n

    async def _prefill_blocks(
        self,
        hashes: List[str],
        rid: str,
        block_priorities: Optional[List[int]] = None,
        block_tags: Optional[List[str]] = None,
        content_hashes: Optional[List[str]] = None,
    ) -> Tuple[float, int, int, int, int]:
        """Walk the chained hashes, simulate prefill for each block.

        Returns (prefill_us, cached_blocks, awaited_blocks, computed_blocks,
        diff_aware_reused_blocks).
        """
        cached = 0
        awaited = 0
        computed = 0
        diff_aware_reused = 0
        us = 0.0
        prefix_broken = False
        for i, h in enumerate(hashes):
            pr = block_priorities[i] if block_priorities and i < len(block_priorities) else 0
            tg = block_tags[i] if block_tags and i < len(block_tags) else ""
            ch = content_hashes[i] if content_hashes and i < len(content_hashes) else ""
            entry = self._cache.get(h)
            now = time.monotonic()
            if entry is None or prefix_broken:
                # Diff-aware cross-iteration reuse: check the content index
                # for a previously-seen block with matching content. If
                # found, charge the RoPE re-application cost only.
                if (
                    self.enable_diff_aware
                    and ch
                    and prefix_broken               # only kicks in past the prefix break
                    and ch in self._content_index
                ):
                    src = self._content_index[ch]
                    src_entry = self._cache.get(src)
                    if src_entry is not None and src_entry.state == "READY":
                        # Install a new entry under this chained hash that
                        # "borrows" from the content-matched block.
                        ev = asyncio.Event(); ev.set()
                        self._cache[h] = _BlockEntry(
                            state="READY", ready_event=ev, last_used_ts=now,
                            priority=max(pr, src_entry.priority),
                            tag=tg or src_entry.tag,
                        )
                        await asyncio.sleep(DIFFAWARE_REUSE_BLOCK_US / 1_000_000.0)
                        us += DIFFAWARE_REUSE_BLOCK_US
                        diff_aware_reused += 1
                        self.stats.diff_aware_hits += 1
                        # Importantly: do NOT set prefix_broken=False — the
                        # CHAIN is still broken from vLLM's perspective; only
                        # this individual block's KV got reused.
                        self.stats.total_blocks_seen += 1
                        self.stats.peak_blocks = max(self.stats.peak_blocks, len(self._cache))
                        continue
                ev = asyncio.Event()
                self._cache[h] = _BlockEntry(
                    state="COMPUTING", ready_event=ev, last_used_ts=now,
                    priority=pr, tag=tg,
                )
                await asyncio.sleep(UNCACHED_BLOCK_US / 1_000_000.0)
                ent = self._cache.get(h)
                if ent is not None:
                    ent.state = "READY"
                    ent.last_used_ts = time.monotonic()
                    ent.ready_event.set()
                # Index this freshly-computed block by its content hash too,
                # so future diff-aware lookups can find it.
                if self.enable_diff_aware and ch:
                    self._content_index[ch] = h
                us += UNCACHED_BLOCK_US
                computed += 1
                self.stats.misses += 1
                prefix_broken = True
            elif entry.state == "READY":
                entry.last_used_ts = now
                # Priority-aware caches respect the *highest* priority anyone
                # ever assigns to a block — once tagged "system", it's hot.
                if pr > entry.priority:
                    entry.priority = pr
                    entry.tag = tg or entry.tag
                us += CACHED_BLOCK_US
                cached += 1
                self.stats.cache_hits += 1
            else:  # COMPUTING
                ev = entry.ready_event
                await ev.wait()
                ent = self._cache.get(h)
                if ent is not None:
                    ent.last_used_ts = time.monotonic()
                    if pr > ent.priority:
                        ent.priority = pr
                        ent.tag = tg or ent.tag
                us += CACHED_BLOCK_US
                awaited += 1
                self.stats.shared_hits += 1
            self.stats.total_blocks_seen += 1
            self.stats.peak_blocks = max(self.stats.peak_blocks, len(self._cache))
        self._evict_if_needed()
        return us, cached, awaited, computed, diff_aware_reused

    # ------------------------------------------------------------------
    # Public chat-completion API
    # ------------------------------------------------------------------

    async def chat_completion(
        self,
        system: str,
        user: str,
        max_output_tokens: int = 200,
        section_priorities: Optional[Dict[str, int]] = None,
    ) -> Tuple[str, CallStats]:
        """Chat completion call.

        section_priorities: optional mapping {"system": int, "user": int}.
        When the AdaEvolve-aware client supplies it, the simulator tags the
        system blocks with priority=3 (system, hot forever) and the user
        blocks with priority=1 (user, evict before system). The
        priority-aware eviction policy uses these tags.
        """
        self._req_counter += 1
        rid = f"req-{self._req_counter:06d}"
        arrived = time.monotonic()

        sys_tokens = fake_tokenize(system)
        user_tokens = fake_tokenize(user)
        full_tokens = sys_tokens + user_tokens
        hashes = chain_hash_blocks(full_tokens)
        content_hashes = (
            content_hash_blocks(full_tokens) if self.enable_diff_aware else None
        )
        sig = hashlib.sha256(("\x1e".join(hashes)).encode()).hexdigest()[:12]

        # Tag each block by which section it predominantly contains.
        sp = section_priorities or {}
        sys_pri = sp.get("system", 3 if sp else 0)
        user_pri = sp.get("user", 1 if sp else 0)
        block_priorities: List[int] = []
        block_tags: List[str] = []
        sys_tok_count = len(sys_tokens)
        for bi in range(len(hashes)):
            block_start = bi * 16
            if block_start + 8 < sys_tok_count:
                block_priorities.append(sys_pri)
                block_tags.append("system")
            else:
                block_priorities.append(user_pri)
                block_tags.append("user")

        wait_start = time.monotonic()
        async with self._prefill_sem:
            queued_us = (time.monotonic() - wait_start) * 1_000_000
            prefill_us, cached, awaited, computed, diff_aware = await self._prefill_blocks(
                hashes, rid,
                block_priorities=block_priorities, block_tags=block_tags,
                content_hashes=content_hashes,
            )

        # Decode (continuous-batched, modeled with a high-cap semaphore).
        async with self._decode_sem:
            decode_us = max_output_tokens * DECODE_US_PER_TOK
            await asyncio.sleep(decode_us / 1_000_000.0)

        finished = time.monotonic()
        text = self._fake_completion(max_output_tokens)

        cs = CallStats(
            request_id=rid,
            arrived_at=arrived - self._t0,
            finished_at=finished - self._t0,
            prompt_tokens=len(full_tokens),
            output_tokens=max_output_tokens,
            total_blocks=len(hashes),
            cached_blocks=cached,
            computed_blocks=computed,
            awaited_blocks=awaited,
            prefill_us=prefill_us,
            decode_us=decode_us,
            queued_us=queued_us,
            prompt_signature=sig,
        )
        self.stats.calls.append(cs)
        self.stats.total_prompt_tokens += len(full_tokens)
        self.stats.total_output_tokens += max_output_tokens
        return text, cs

    def _fake_completion(self, n_tokens: int) -> str:
        # Returns a parseable-looking code block so a real parser would
        # accept it — we don't actually parse in the bench, but keeping
        # this realistic helps if anyone wires the mock into the real
        # controller.
        words = ["x", "y", "z", "a", "b", "c", "k", "m", "n", "p"]
        body = " ".join(self._rng.choice(words) for _ in range(n_tokens))
        return f"```python\n# generated\n{body}\n```"


# ----------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------


def summarize_global(stats: GlobalStats) -> Dict[str, float]:
    n = max(1, len(stats.calls))
    durations_ms = sorted((c.finished_at - c.arrived_at) * 1000.0 for c in stats.calls)
    p = lambda q: durations_ms[min(int(q * (len(durations_ms) - 1)), len(durations_ms) - 1)]

    total_lookups = stats.cache_hits + stats.shared_hits + stats.misses
    direct_hit_rate = stats.cache_hits / max(1, total_lookups)
    full_hit_rate = (stats.cache_hits + stats.shared_hits) / max(1, total_lookups)
    saved_us = stats.cache_hits * (UNCACHED_BLOCK_US - CACHED_BLOCK_US) + stats.shared_hits * (
        UNCACHED_BLOCK_US - CACHED_BLOCK_US
    )
    return {
        "calls": n,
        "prompt_tokens_total": stats.total_prompt_tokens,
        "output_tokens_total": stats.total_output_tokens,
        "total_blocks_seen": stats.total_blocks_seen,
        "cache_hits": stats.cache_hits,
        "shared_hits": stats.shared_hits,
        "misses": stats.misses,
        "direct_hit_rate": direct_hit_rate,
        "effective_hit_rate": full_hit_rate,
        "evictions": stats.evictions,
        "peak_blocks": stats.peak_blocks,
        "prefill_us_saved_vs_no_cache": saved_us,
        "lat_p50_ms": p(0.5),
        "lat_p90_ms": p(0.9),
        "lat_p99_ms": p(0.99),
        "wall_time_s": max(c.finished_at for c in stats.calls) if stats.calls else 0.0,
    }
