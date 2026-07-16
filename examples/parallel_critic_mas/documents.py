"""Reuses the evidence_heavy_mas document corpus.

Kept separate so this example can be understood standalone without importing
from sibling example dirs (which pytest / notebook execution may not have on
sys.path)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Document:
    id: str
    label: str
    text: str


CORPUS: list[Document] = [
    Document(
        id="tcp-primer",
        label="TCP: connection-oriented reliable delivery",
        text=(
            "TCP (Transmission Control Protocol) is connection-oriented. Before "
            "any data flows, endpoints complete a three-way handshake (SYN, "
            "SYN-ACK, ACK). Once established, TCP guarantees that bytes are "
            "delivered in order, retransmits lost segments, and applies flow "
            "and congestion control. TCP headers are at least 20 bytes."
        ),
    ),
    Document(
        id="udp-primer",
        label="UDP: connectionless best-effort datagrams",
        text=(
            "UDP is connectionless. It sends datagrams without establishing a "
            "connection and without delivery, ordering, or duplicate-protection "
            "guarantees. UDP headers are 8 bytes. Applications that prioritize "
            "low latency over reliability, such as DNS, real-time voice/video, "
            "and online gaming, use UDP."
        ),
    ),
    Document(
        id="dns-primer",
        label="DNS: name resolution over UDP (mostly)",
        text=(
            "DNS resolves human-readable names to IP addresses. Most queries "
            "use UDP on port 53 for low latency; responses larger than ~512 "
            "bytes fall back to TCP. DNSSEC adds cryptographic signatures over "
            "records to detect tampering."
        ),
    ),
    Document(
        id="tls-primer",
        label="TLS: encrypted transport atop TCP",
        text=(
            "TLS provides confidentiality and integrity for data in transit. "
            "It runs on top of TCP. TLS 1.3 reduces handshake round-trips to "
            "one RTT (or zero for session resumption)."
        ),
    ),
    Document(
        id="http2-primer",
        label="HTTP/2: multiplexing over one TCP connection",
        text=(
            "HTTP/2 introduces binary framing and multiplexes many concurrent "
            "requests over one TCP connection. It runs almost universally over "
            "TLS in practice; the ALPN extension negotiates the protocol "
            "version during the TLS handshake."
        ),
    ),
]


def render_corpus() -> str:
    return "\n\n".join(f"[{d.id}] {d.label}\n{d.text}" for d in CORPUS)
