import functools
import os
from pathlib import Path
from typing import List
import torch
import numpy as np
from sentence_transformers import SentenceTransformer

from .models import PluginIn, PluginWithScore, TaskIn, TaskWithScore
from .storage import PluginFromDB, TaskFromDB, TaskStorage

DEFAULT_MODEL_NAME = "BAAI/bge-m3"


def _resolve_model_path() -> str:
    local_override = os.getenv("BGE_M3_MODEL_PATH")
    if local_override:
        return str(Path(local_override).expanduser())

    custom_model = os.getenv("BGE_M3_MODEL")
    if custom_model:
        return custom_model

    return DEFAULT_MODEL_NAME


@functools.lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    model_id = _resolve_model_path()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(model_id, device=device)


def warmup_model() -> SentenceTransformer:
    return _load_model()


def encode_text(text: str) -> np.ndarray:
    embedding = _load_model().encode(
        text, convert_to_numpy=True, normalize_embeddings=True
    )
    if embedding.ndim > 1:
        embedding = embedding[0]
    return embedding.astype(np.float32)


def cosine_similarity(query_vec: np.ndarray, doc_vec: np.ndarray) -> float:
    if query_vec.size == 0 or doc_vec.size == 0:
        return 0.0
    return float(np.dot(query_vec, doc_vec))


class SemanticTaskSearchService:
    def __init__(self, storage: TaskStorage):
        self.storage = storage

    def index_task(self, task: TaskIn) -> None:
        text = "\n".join(
            filter(None, [task.name, task.description or "", task.category or ""])
        )
        embedding = encode_text(text)
        self.storage.upsert_task(task, embedding)

    def bulk_index(self, tasks: List[TaskIn]) -> None:
        for task in tasks:
            self.index_task(task)

    def search(self, query: str, top_k: int = 3) -> List[TaskWithScore]:
        if not query or not query.strip():
            return []

        query_vec = encode_text(query)
        tasks: List[TaskFromDB] = self.storage.get_all_tasks()

        scored_tasks: List[TaskWithScore] = []
        for task in tasks:
            score = cosine_similarity(query_vec, task.embedding)
            scored_tasks.append(
                TaskWithScore(
                    id=task.id,
                    name=task.name,
                    description=task.description,
                    category=task.category,
                    score=score,
                )
            )

        scored_tasks.sort(key=lambda t: t.score, reverse=True)
        return scored_tasks[: max(top_k, 0)]


class SemanticPluginSearchService:
    def __init__(self, storage: TaskStorage):
        self.storage = storage

    def index_plugin(self, plugin: PluginIn) -> None:
        text = "\n".join(
            filter(
                None,
                [
                    plugin.name,
                    plugin.description or "",
                    plugin.category or "",
                    plugin.graphType or "",
                    plugin.pluginType or "",
                ],
            )
        )
        embedding = encode_text(text)
        self.storage.upsert_plugin(plugin, embedding)

    def bulk_index(self, plugins: List[PluginIn]) -> None:
        for plugin in plugins:
            self.index_plugin(plugin)

    def search(self, query: str, top_k: int = 3) -> List[PluginWithScore]:
        if not query or not query.strip():
            return []

        query_vec = encode_text(query)
        plugins: List[PluginFromDB] = self.storage.get_all_plugins()

        scored_plugins: List[PluginWithScore] = []
        for plugin in plugins:
            score = cosine_similarity(query_vec, plugin.embedding)
            scored_plugins.append(
                PluginWithScore(
                    id=plugin.id,
                    name=plugin.name,
                    description=plugin.description,
                    category=plugin.category,
                    graphType=plugin.graph_type,
                    pluginType=plugin.plugin_type,
                    score=score,
                )
            )

        scored_plugins.sort(key=lambda p: p.score, reverse=True)
        return scored_plugins[: max(top_k, 0)]
