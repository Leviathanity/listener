import aiosqlite
from datetime import datetime, timezone


SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    file_path TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    progress REAL DEFAULT 0.0,
    progress_detail TEXT,
    result_path TEXT,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


class TaskTracker:
    def __init__(self, db_path: str):
        self._db_path = db_path
        self._conn = None

    async def init(self):
        self._conn = await aiosqlite.connect(self._db_path)
        self._conn.row_factory = aiosqlite.Row
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

    async def close(self):
        if self._conn:
            await self._conn.close()

    async def __aenter__(self):
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        return False

    async def create(self, task_id: str, filename: str, file_path: str):
        now = datetime.now(timezone.utc).isoformat()
        await self._conn.execute(
            "INSERT INTO tasks (id, filename, file_path, status, created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?)",
            (task_id, filename, file_path, now, now),
        )
        await self._conn.commit()

    async def get(self, task_id: str) -> dict | None:
        cursor = await self._conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def update(self, task_id: str, **kwargs):
        allowed = {"status", "progress", "progress_detail", "result_path", "error_message"}
        unknown = set(kwargs) - allowed
        if unknown:
            raise ValueError(f"Unknown fields: {unknown}")
        if not kwargs:
            return
        kwargs = dict(kwargs)
        kwargs["updated_at"] = datetime.now(timezone.utc).isoformat()
        set_clause = ", ".join(f"{k} = ?" for k in kwargs)
        values = list(kwargs.values()) + [task_id]
        await self._conn.execute(f"UPDATE tasks SET {set_clause} WHERE id = ?", values)
        await self._conn.commit()

    async def count_processing(self) -> int:
        cursor = await self._conn.execute("SELECT COUNT(*) as cnt FROM tasks WHERE status = 'processing'")
        row = await cursor.fetchone()
        return row["cnt"]
