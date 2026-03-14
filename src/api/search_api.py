from __future__ import annotations

from pathlib import Path
from typing import Any, List

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

from src.main import SearchSystem, load_config


ML_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_CACHE: dict[str, Any] | None = None


class SearchRequest(BaseModel):
    query: str
    top_k: int | None = None


class SearchResult(BaseModel):
    cte_id: int
    score: float


class SearchResponse(BaseModel):
    results: List[SearchResult]


app = FastAPI(title="TenderHack Search API")

_search_system: SearchSystem | None = None
_default_top_k: int = 10


def _init_search_system() -> SearchSystem:
    global _search_system, _CONFIG_CACHE, _default_top_k  # noqa: PLW0603
    if _search_system is not None:
        return _search_system

    if _CONFIG_CACHE is None:
        _CONFIG_CACHE = load_config()

    config: dict[str, Any] = _CONFIG_CACHE

    prod_model_name = config["prod_model_name"]
    prod_index_rel = config["prod_index_path"]
    _default_top_k = int(config.get("search_top_k", 10))

    index_path = ML_ROOT / prod_index_rel
    emb_dir = ML_ROOT / config["embeddings_dir"]
    ids_path = emb_dir / "cte_ids.npy"

    _search_system = SearchSystem(
        model_name=prod_model_name,
        index_path=index_path,
        ids_path=ids_path,
        available_models=config.get("available_models", []),
    )
    return _search_system


@app.on_event("startup")
def on_startup() -> None:
    _init_search_system()


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    system = _init_search_system()
    top_k = req.top_k if req.top_k is not None else _default_top_k
    cte_ids, scores = system.search(req.query, top_k=top_k)

    results = [
        SearchResult(cte_id=int(cid), score=float(s))
        for cid, s in zip(cte_ids, scores)
    ]
    return SearchResponse(results=results)

