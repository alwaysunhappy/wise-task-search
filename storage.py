import sqlite3
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import List

import numpy as np

from .models import PluginIn, TaskIn


@dataclass
class TaskFromDB:
    id: str
    name: str
    description: str
    category: str
    embedding: np.ndarray


@dataclass
class PluginFromDB:
    id: str
    name: str
    description: str
    category: str
    graph_type: str
    plugin_type: str
    embedding: np.ndarray


class TaskStorage:
    def __init__(self, db_path: str = "semantic_search.db"):
        base_path = Path(__file__).resolve().parent
        self.db_path = base_path / db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute("PRAGMA journal_mode = WAL;")
        return conn

    def init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS plugins (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    description TEXT,
                    category TEXT,
                    graph_type TEXT,
                    plugin_type TEXT,
                    embedding BLOB NOT NULL
                )
                """
            )
            conn.commit()

    def upsert_task(self, task: TaskIn, embedding: np.ndarray) -> None:
        embedding_blob = self._serialize_embedding(embedding)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tasks (id, name, description, category, embedding)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    description=excluded.description,
                    category=excluded.category,
                    embedding=excluded.embedding
                """,
                (
                    task.id,
                    task.name,
                    task.description or "",
                    task.category or "",
                    embedding_blob,
                ),
            )
            conn.commit()

    def upsert_plugin(self, plugin: PluginIn, embedding: np.ndarray) -> None:
        embedding_blob = self._serialize_embedding(embedding)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO plugins (id, name, description, category, graph_type, plugin_type, embedding)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    name=excluded.name,
                    description=excluded.description,
                    category=excluded.category,
                    graph_type=excluded.graph_type,
                    plugin_type=excluded.plugin_type,
                    embedding=excluded.embedding
                """,
                (
                    plugin.id,
                    plugin.name,
                    plugin.description or "",
                    plugin.category or "",
                    plugin.graphType or "",
                    plugin.pluginType or "",
                    embedding_blob,
                ),
            )
            conn.commit()

    def get_all_tasks(self) -> List[TaskFromDB]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT id, name, description, category, embedding FROM tasks"
            )
            rows = cursor.fetchall()

        tasks: List[TaskFromDB] = []
        for row in rows:
            embedding = self._deserialize_embedding(row["embedding"])
            tasks.append(
                TaskFromDB(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"] or "",
                    category=row["category"] or "",
                    embedding=embedding,
                )
            )
        return tasks

    def get_all_plugins(self) -> List[PluginFromDB]:
        with self._connect() as conn:
            cursor = conn.execute(
                "SELECT id, name, description, category, graph_type, plugin_type, embedding FROM plugins"
            )
            rows = cursor.fetchall()

        plugins: List[PluginFromDB] = []
        for row in rows:
            embedding = self._deserialize_embedding(row["embedding"])
            plugins.append(
                PluginFromDB(
                    id=row["id"],
                    name=row["name"],
                    description=row["description"] or "",
                    category=row["category"] or "",
                    graph_type=row["graph_type"] or "",
                    plugin_type=row["plugin_type"] or "",
                    embedding=embedding,
                )
            )
        return plugins

    def _serialize_embedding(self, embedding: np.ndarray) -> bytes:
        buffer = BytesIO()
        np.save(buffer, embedding.astype(np.float32))
        return buffer.getvalue()

    def _deserialize_embedding(self, blob: bytes) -> np.ndarray:
        buffer = BytesIO(blob)
        buffer.seek(0)
        return np.load(buffer)
