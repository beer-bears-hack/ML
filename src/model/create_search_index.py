import argparse
import sys
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import yaml


# Script lives at ML/src/model/create_search_index.py -> ML root is parents[2]
ML_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ML_ROOT / "config" / "ml_config.yaml"


def load_config() -> dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found at {CONFIG_PATH}. "
            "Ensure ML/config/ml_config.yaml exists."
        )
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a FAISS search index from precomputed embeddings (.npy) "
            "and save it to ML/models/indexes/."
        )
    )
    parser.add_argument(
        "-e",
        "--embeddings-path",
        required=True,
        help=(
            "Path to .npy file with embeddings. "
            "If relative, interpreted relative to ML project root."
        ),
    )
    parser.add_argument(
        "-t",
        "--index-type",
        choices=("flat", "ivf"),
        default="flat",
        help="FAISS index type: 'flat' (exact) or 'ivf' (ANN via IVF-Flat).",
    )
    parser.add_argument(
        "--nlist",
        type=int,
        default=1024,
        help=(
            "Number of clusters (lists) for IVF index. "
            "Used only when --index-type=ivf."
        ),
    )
    parser.add_argument(
        "--nprobe",
        type=int,
        default=64,
        help=(
            "Number of clusters to search over for IVF index. "
            "Used only when --index-type=ivf."
        ),
    )
    return parser.parse_args()


def resolve_paths(config: dict[str, Any], embeddings_path_arg: str) -> tuple[Path, Path]:
    """
    Return (embeddings_path, indexes_dir). Paths are resolved relative to ML root.
    """
    emb_path = Path(embeddings_path_arg)
    if not emb_path.is_absolute():
        emb_path = ML_ROOT / emb_path

    indexes_dir_cfg = config.get("indexes_dir", "models/indexes")
    indexes_dir = ML_ROOT / indexes_dir_cfg
    return emb_path, indexes_dir


def l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return x / norms


def infer_model_short_name(embeddings_path: Path) -> str:
    """
    Infer a short model identifier from the embeddings file name.

    Example:
        models/embeddings/intfloat_multilingual-e5-small.npy
        -> intfloat_multilingual-e5-small
    """
    stem = embeddings_path.stem
    if stem.startswith("test_"):
        stem = stem[len("test_") :]
    return stem


def build_flat_index(vectors: np.ndarray) -> faiss.Index:
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index


def build_ivf_index(vectors: np.ndarray, nlist: int, nprobe: int) -> faiss.IndexIVFFlat:
    dim = vectors.shape[1]
    nlist = max(1, nlist)
    nprobe = max(1, min(nprobe, nlist))

    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)

    index.train(vectors)
    index.add(vectors)
    index.nprobe = nprobe
    return index


def build_and_save_index(
    embeddings_path: Path,
    indexes_dir: Path,
    index_type: str,
    nlist: int,
    nprobe: int,
) -> None:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    print(f"ML root: {ML_ROOT}")
    print(f"Loading embeddings from: {embeddings_path}")

    vectors = np.load(embeddings_path)
    if vectors.ndim != 2:
        raise ValueError(
            f"Expected 2D array of embeddings, got shape {vectors.shape!r}."
        )

    if vectors.dtype != np.float32:
        vectors = vectors.astype("float32")

    print(f"Loaded embeddings with shape: {vectors.shape}, dtype: {vectors.dtype}")

    vectors = l2_normalize(vectors)
    print("Applied L2 normalization to embeddings (for cosine/IP search).")

    if index_type == "flat":
        print("Building FAISS IndexFlatIP (exact search).")
        index = build_flat_index(vectors)
        params_str = "metric=ip"
    else:
        print(
            f"Building FAISS IndexIVFFlat (ANN) with nlist={nlist}, nprobe={nprobe}."
        )
        index = build_ivf_index(vectors, nlist=nlist, nprobe=nprobe)
        params_str = f"nlist={nlist}_nprobe={nprobe}"

    indexes_dir.mkdir(parents=True, exist_ok=True)

    model_short = infer_model_short_name(embeddings_path)
    filename = f"faiss_{index_type}_({params_str})__{model_short}.index"
    index_path = indexes_dir / filename

    faiss.write_index(index, str(index_path))
    print(f"Saved FAISS index to: {index_path}")


def main() -> None:
    try:
        config = load_config()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    args = parse_args()

    try:
        embeddings_path, indexes_dir = resolve_paths(config, args.embeddings_path)
        build_and_save_index(
            embeddings_path=embeddings_path,
            indexes_dir=indexes_dir,
            index_type=args.index_type,
            nlist=args.nlist,
            nprobe=args.nprobe,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

