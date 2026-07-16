"""Same 5-task benchmark as evidence_heavy_mas (kept separate for standalone use)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    id: str
    difficulty: str
    question: str
    must_mention: tuple[str, ...]


TASKS: list[Task] = [
    Task("easy-tcp-header-size", "easy",
         "What is the minimum TCP header size according to the provided documents?",
         ("20",)),
    Task("easy-udp-use-cases", "easy",
         "Name two example applications from the documents that use UDP rather than TCP.",
         ("DNS",)),
    Task("medium-dns-fallback", "medium",
         "According to the documents, when does DNS use TCP instead of UDP, and why does that matter?",
         ("512", "TCP")),
    Task("medium-tls-layers", "medium",
         "Where does TLS sit relative to TCP, and what does TLS 1.3 change about the handshake?",
         ("TCP", "RTT")),
    Task("hard-http2-tls-alpn", "hard",
         "Explain the layering when a browser fetches a page via HTTP/2 over TLS: which layer handles reliability, where multiplexing sits, and how the protocol version is negotiated.",
         ("TCP", "TLS", "ALPN", "multiplex")),
]


def score_answer(task: Task, answer: str) -> tuple[float, dict[str, float], str]:
    lowered = answer.lower()
    hits = sum(1 for s in task.must_mention if s.lower() in lowered)
    coverage = hits / max(1, len(task.must_mention))
    axes = {"coverage": coverage}
    reasoning = f"rubric: {hits}/{len(task.must_mention)} required facts covered"
    return coverage, axes, reasoning
