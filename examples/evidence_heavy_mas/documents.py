"""Small in-code document corpus for the evidence_heavy MAS example.

Five short primers on core internet protocols. Each has an id, a short label,
and body text. The MAS's memory node retrieves relevant snippets to ground the
solver's answer. Kept in-code so the example has zero filesystem I/O.
"""

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
            "and congestion control. These guarantees add per-message overhead "
            "and typically some latency. TCP headers are at least 20 bytes."
        ),
    ),
    Document(
        id="udp-primer",
        label="UDP: connectionless best-effort datagrams",
        text=(
            "UDP (User Datagram Protocol) is connectionless. It sends datagrams "
            "without establishing a connection and without delivery, ordering, "
            "or duplicate-protection guarantees. UDP headers are 8 bytes. "
            "Applications that prioritize low latency over reliability, such as "
            "DNS lookups, real-time voice/video, and online gaming, use UDP."
        ),
    ),
    Document(
        id="dns-primer",
        label="DNS: name resolution over UDP (mostly)",
        text=(
            "DNS (Domain Name System) resolves human-readable names to IP "
            "addresses. Most queries use UDP on port 53 for low latency; "
            "responses larger than ~512 bytes fall back to TCP. DNS records "
            "have TTL (time-to-live) values that resolvers use to cache "
            "responses. DNSSEC adds cryptographic signatures over records to "
            "detect tampering, but does not encrypt the query itself."
        ),
    ),
    Document(
        id="tls-primer",
        label="TLS: encrypted transport atop TCP",
        text=(
            "TLS (Transport Layer Security) provides confidentiality and "
            "integrity for data in transit. It runs on top of TCP. The handshake "
            "negotiates a cipher suite and establishes session keys via "
            "asymmetric cryptography; bulk data then uses fast symmetric "
            "encryption. TLS 1.3 reduces handshake round-trips to one RTT (or "
            "zero for session resumption) compared to prior versions."
        ),
    ),
    Document(
        id="http2-primer",
        label="HTTP/2: multiplexing over one TCP connection",
        text=(
            "HTTP/2 introduces binary framing and multiplexes many concurrent "
            "requests over one TCP connection, eliminating head-of-line blocking "
            "at the application layer. It supports server push and header "
            "compression via HPACK. HTTP/2 runs almost universally over TLS in "
            "practice; the ALPN extension negotiates the protocol version "
            "during the TLS handshake."
        ),
    ),
]


DOCS_BY_ID = {d.id: d for d in CORPUS}


def render_corpus() -> str:
    """Return a plain-text bundle of the whole corpus, ready to paste into a
    system prompt. Used by the memory node when doing simple retrieval."""
    return "\n\n".join(f"[{d.id}] {d.label}\n{d.text}" for d in CORPUS)
