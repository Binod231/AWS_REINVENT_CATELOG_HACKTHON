"""Local catalog cache, full-text search, and change detection.

Three reasons this layer exists, in order of importance:

1. **The API does not search.** The developer guide is explicit: "It does not
   search or filter the catalog. You retrieve an event's sessions and filter them
   in your own application." So a local index is not an optimisation, it is the
   only way to answer "show me the 300-level Bedrock sessions on Wednesday".

2. **Quotas are per minute.** ListSessions allows 120 requests a minute. Walking
   a multi-thousand-session catalog on every command would burn that allowance and
   make the tool unusable. One sync, many queries.

3. **The catalog moves, and the API has no change feed.** Sessions get added,
   rooms get reassigned, times shift, and seats fill. There is no ``updated_at``
   and no ETag to ask about it. Keeping the previous sync lets replan compute the
   diff itself and tell an attendee what changed since they last looked - which is
   information the official app does not give them.
"""

from __future__ import annotations

from replan.cache.changes import (
    ChangeKind,
    FieldChange,
    SessionChange,
    SyncDiff,
    session_fingerprint,
)
from replan.cache.store import (
    CacheInfo,
    CatalogCache,
    SearchFilters,
    SearchHit,
    build_fts_query,
)

__all__ = [
    "CacheInfo",
    "CatalogCache",
    "ChangeKind",
    "FieldChange",
    "SearchFilters",
    "SearchHit",
    "SessionChange",
    "SyncDiff",
    "build_fts_query",
    "session_fingerprint",
]
