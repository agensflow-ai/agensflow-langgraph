"""Test-time env setup — force server to use in-memory SQLite so integration tests
don't need a `./data/` directory."""

from __future__ import annotations

import os

os.environ.setdefault("AGF_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("AGF_ENV", "test")
os.environ.setdefault("AGF_JWT_SECRET", "test-secret-not-for-production")
