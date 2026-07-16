"""AgensFlow adapter for LangGraph.

Public surface:
    @agensflow(pool={...})     # decorate any LangGraph node
    record_reward(...)         # explicit reward at graph terminal
    arecord_reward(...)        # async variant
    import_policy(...)         # warm-start from an exported policy JSON
    get_client(...)            # direct HTTP client if you need finer control
"""

from __future__ import annotations

from agensflow_langgraph.client import get_client
from agensflow_langgraph.decorator import agensflow
from agensflow_langgraph.errors import (
    AgensFlowError,
    InvalidPool,
    ServerRejected,
    ServerUnreachable,
)
from agensflow_langgraph.sidecar import arecord_reward, record_reward
from agensflow_langgraph.warm_start import export_policy, import_policy

__version__ = "0.1.0"

__all__ = [
    "agensflow",
    "record_reward",
    "arecord_reward",
    "import_policy",
    "export_policy",
    "get_client",
    "AgensFlowError",
    "ServerUnreachable",
    "ServerRejected",
    "InvalidPool",
    "__version__",
]
