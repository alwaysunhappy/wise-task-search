from dataclasses import field
from typing import List, Optional

from pydantic.dataclasses import dataclass


@dataclass
class TaskIn:
    id: str
    name: str
    description: Optional[str] = ""
    category: Optional[str] = ""


@dataclass
class TaskWithScore:
    id: str
    name: str
    score: float
    description: Optional[str] = ""
    category: Optional[str] = ""


@dataclass
class PluginIn:
    id: str
    name: str
    description: Optional[str] = ""
    category: Optional[str] = ""
    graphType: Optional[str] = ""
    pluginType: Optional[str] = ""


@dataclass
class PluginWithScore:
    id: str
    name: str
    score: float
    description: Optional[str] = ""
    category: Optional[str] = ""
    graphType: Optional[str] = ""
    pluginType: Optional[str] = ""


@dataclass
class BulkIndexRequest:
    tasks: List[TaskIn] = field(default_factory=list)


@dataclass
class PluginBulkIndexRequest:
    plugins: List[PluginIn] = field(default_factory=list)


@dataclass
class SearchRequest:
    query: str
    top_k: int = 3


@dataclass
class PluginSearchRequest:
    query: str
    top_k: int = 3
