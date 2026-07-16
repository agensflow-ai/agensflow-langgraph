"""The 5-task benchmark set with difficulty labels + judge rubrics.

Each task has a difficulty (easy/medium/hard) and a set of expected-answer
signals — key facts the final answer must include and stop-flags for
disallowed content. The judge scores against these signals; the substrate
learns to route solver calls to appropriately-sized models per difficulty.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    difficulty: str  # "easy" | "medium" | "hard"
    question: str
    must_mention: tuple[str, ...]  # substrings the final answer should contain
    must_not_claim: tuple[str, ...] = ()  # substrings that would indicate an error


TASKS: list[Task] = [
    # Easy — single-doc factual lookups. Cheap models should suffice.
    Task(
        id="easy-tcp-header-size",
        difficulty="easy",
        question=(
            "What is the minimum TCP header size according to the provided documents? "
            "Answer with just the byte count in a single short sentence."
        ),
        must_mention=("20",),
    ),
    Task(
        id="easy-udp-use-cases",
        difficulty="easy",
        question=(
            "Name two example applications from the provided documents that use "
            "UDP rather than TCP."
        ),
        must_mention=("DNS",),  # allow either DNS/voice/video/gaming — DNS is universal
    ),
    # Medium — synthesizing two documents. Balanced models should win here.
    Task(
        id="medium-dns-fallback",
        difficulty="medium",
        question=(
            "According to the provided documents, when does DNS use TCP instead "
            "of UDP, and why does that matter for latency?"
        ),
        must_mention=("512", "TCP"),
    ),
    Task(
        id="medium-tls-layers",
        difficulty="medium",
        question=(
            "Where does TLS sit relative to TCP, according to the documents, "
            "and what does TLS 1.3 change about the handshake compared to prior "
            "versions?"
        ),
        must_mention=("TCP", "RTT"),
    ),
    # Hard — multi-hop reasoning across TCP, TLS, and HTTP/2. Deep models needed.
    Task(
        id="hard-http2-tls-alpn",
        difficulty="hard",
        question=(
            "Using ONLY the provided documents, explain the layering when a "
            "browser fetches a page via HTTP/2 over TLS. Cover: which layer "
            "handles reliability, where multiplexing sits, and how the protocol "
            "version is negotiated. Answer in 3-4 sentences."
        ),
        must_mention=("TCP", "TLS", "ALPN", "multiplex"),
    ),
]


TASKS_BY_ID = {t.id: t for t in TASKS}


def score_answer(task: Task, answer: str) -> tuple[float, dict[str, float], str]:
    """Rule-based rubric score for the final answer.

    Returns (quality, axes, reasoning). Quality is the fraction of `must_mention`
    substrings present (case-insensitive), penalized if any `must_not_claim`
    substrings appear. Axes give the per-signal breakdown so the AgensFlow
    dashboard can show WHY a score was assigned. No LLM judge required — the
    rubric is deterministic so the demo has no per-run judge cost.
    """
    lowered = answer.lower()
    hits = sum(1 for s in task.must_mention if s.lower() in lowered)
    forbidden = sum(1 for s in task.must_not_claim if s.lower() in lowered)
    coverage = hits / max(1, len(task.must_mention))
    penalty = 0.5 * min(1.0, forbidden / max(1, len(task.must_not_claim) or 1))
    quality = max(0.0, coverage - penalty)
    axes = {
        "coverage": coverage,
        "no_forbidden_claims": 1.0 - penalty,
    }
    reasoning = (
        f"rubric: {hits}/{len(task.must_mention)} required facts covered"
        + (f"; {forbidden} forbidden claims" if task.must_not_claim else "")
    )
    return quality, axes, reasoning
