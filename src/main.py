from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, List, Tuple

import faiss
import numpy as np
import torch
import yaml
from sentence_transformers import SentenceTransformer


# Script lives at ML/src/main.py -> ML root is parents[1]
ML_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ML_ROOT / "config" / "ml_config.yaml"


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found at {CONFIG_PATH}. "
            "Ensure ML/config/ml_config.yaml exists."
        )
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_model_name(model_name: str, available_models: List[str]) -> None:
    if model_name not in available_models:
        msg = (
            f"Unsupported model '{model_name}'. "
            f"Supported models are: {', '.join(available_models)}"
        )
        raise ValueError(msg)


def parse_args(config: dict[str, Any]) -> argparse.Namespace:
    available = config.get("available_models", [])
    parser = argparse.ArgumentParser(
        description="Run semantic search over CTE items using FAISS index.",
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Embedding model name. Must be one of: " + ", ".join(available),
    )
    parser.add_argument(
        "-i",
        "--index-path",
        required=True,
        help=(
            "Path to FAISS index file. "
            "If relative, interpreted relative to ML project root."
        ),
    )
    parser.add_argument(
        "--ids-path",
        help=(
            "Optional path to .npy file with CTE ids. "
            "If not provided, derived from config['embeddings_dir'] "
            "(cte_ids.npy or test_cte_ids.npy)."
        ),
    )
    parser.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=10,
        help="Number of nearest neighbours to return.",
    )
    parser.add_argument(
        "-q",
        "--query",
        required=True,
        help="Text query to search similar CTE items for.",
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Use test ids file (test_cte_ids.npy) instead of cte_ids.npy.",
    )
    return parser.parse_args()


def resolve_paths(
    config: dict[str, Any],
    index_path_arg: str,
    ids_path_arg: str | None,
    test_mode: bool,
) -> Tuple[Path, Path]:
    """
    Return (index_path, ids_path). Paths are resolved relative to ML root.
    """
    index_path = Path(index_path_arg)
    if not index_path.is_absolute():
        index_path = ML_ROOT / index_path

    if ids_path_arg is not None:
        ids_path = Path(ids_path_arg)
        if not ids_path.is_absolute():
            ids_path = ML_ROOT / ids_path
    else:
        emb_dir = ML_ROOT / config["embeddings_dir"]
        fname = "test_cte_ids.npy" if test_mode else "cte_ids.npy"
        ids_path = emb_dir / fname

    return index_path, ids_path


def prepare_text_for_model(model_name: str, text: str) -> str:
    lower_name = model_name.lower()
    if "e5" in lower_name:
        return "query: " + text
    return text


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


class SearchSystem:
    def __init__(
        self,
        model_name: str,
        index_path: Path,
        ids_path: Path,
        available_models: List[str],
    ) -> None:
        validate_model_name(model_name, available_models)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = model_name
        self.model = SentenceTransformer(
            model_name,
            trust_remote_code=True,
            device=device,
        )
        print(f"Initialized model: {model_name} on {device}")

        if not index_path.exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
        self.index = faiss.read_index(str(index_path))
        print(f"Loaded FAISS index from: {index_path}")

        if not ids_path.exists():
            raise FileNotFoundError(f"CTE ids file not found: {ids_path}")
        self.cte_ids = np.load(ids_path)
        print(f"Loaded CTE ids from: {ids_path} (shape={self.cte_ids.shape})")

    def encode_query(self, query: str) -> np.ndarray:
        prepared = prepare_text_for_model(self.model_name, query)
        embeddings = self.model.encode(
            [prepared],
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
        embeddings = l2_normalize(embeddings)
        return embeddings

    def search(self, query: str, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        vec = self.encode_query(query)
        scores, indices = self.index.search(vec, top_k)
        scores = scores[0]
        indices = indices[0]
        matched_ids = self.cte_ids[indices]
        return matched_ids, scores


def main() -> None:
    try:
        config = load_config()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    args = parse_args(config)

    try:
        index_path, ids_path = resolve_paths(
            config=config,
            index_path_arg=args.index_path,
            ids_path_arg=args.ids_path,
            test_mode=args.test,
        )

        system = SearchSystem(
            model_name=args.model,
            index_path=index_path,
            ids_path=ids_path,
            available_models=config.get("available_models", []),
        )

        matched_ids, scores = system.search(args.query, top_k=args.top_k)

        print("Top results (cte_id, score):")
        for cid, s in zip(matched_ids, scores):
            print(f"{cid}\t{s:.4f}")

    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
