---
title: Administration
weight: 21
---

Qdrant exposes administration tools which enable to modify at runtime the behavior of a running instance without changing its configuration manually.

## Locking

Qdrant has locking functionality when you want to turn off some features from qdrant process. Locking is not persistent and you have to lock again after restarting qdrant.

Lock request sample:

```http
POST /locks
{
    "error_message": "write is forbidden",
    "write": true
}
```

Write flags enables/disables write lock.
If write lock is enabled, qdrant doesn't allow creating collections or adding new data to storage.
But deletion or updating is not forbidden.
It is useful when qdrant uses too much disk space and administrator wants to limit disk usage.

You can optionally provide the error message that should be used for error responses to users.
