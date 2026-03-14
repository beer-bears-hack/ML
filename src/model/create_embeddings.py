import argparse
import sys
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd
import torch
import yaml
from sentence_transformers import SentenceTransformer

# Script lives at ML/src/model/create_embeddings.py -> ML root is parents[2]
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


def parse_args(config: dict[str, Any]) -> argparse.Namespace:
    available = config.get("available_models", [])
    parser = argparse.ArgumentParser(
        description=(
            "Build CTE embeddings for a given model and save them to "
            "ML/models/embeddings/."
        )
    )
    parser.add_argument(
        "-m",
        "--model",
        required=True,
        help="Embedding model name. Must be one of: " + ", ".join(available),
    )
    parser.add_argument(
        "-t",
        "--test",
        action="store_true",
        help="Test mode: use only the first 100 rows of CTE.csv.",
    )
    return parser.parse_args()


def validate_model_name(model_name: str, available_models: List[str]) -> None:
    if model_name not in available_models:
        msg = (
            f"Unsupported model '{model_name}'. "
            f"Supported models are: {', '.join(available_models)}"
        )
        raise ValueError(msg)


def resolve_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    """
    Return (cte_path, embeddings_dir). Paths are relative to ML root.
    """
    cte_path = ML_ROOT / config["cte_csv_path"]
    embeddings_dir = ML_ROOT / config["embeddings_dir"]
    return cte_path, embeddings_dir


def build_item_text(row: pd.Series) -> str:
    parts: list[str] = []
    if pd.notna(row.get("CTE_name")):
        parts.append(f"Наименование: {row['CTE_name']}")
    if pd.notna(row.get("category")):
        parts.append(f"Категория: {row['category']}")
    if pd.notna(row.get("manufacturer")):
        parts.append(f"Производитель: {row['manufacturer']}")
    if pd.notna(row.get("characteristics")):
        parts.append(f"Характеристики: {row['characteristics']}")
    return ". ".join(parts)


def prepare_texts_for_model(
    model_name: str,
    texts: list[str],
) -> list[str]:
    lower_name = model_name.lower()
    if "e5" in lower_name:
        prefix = "passage: "
        return [prefix + t for t in texts]
    return texts


def encode_texts(
    model_name: str,
    texts: list[str],
    device: str,
    batch_size: int = 64,
) -> np.ndarray:
    model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
    prepared = prepare_texts_for_model(model_name, texts)

    embeddings: list[np.ndarray] = []
    for i in range(0, len(prepared), batch_size):
        batch = prepared[i : i + batch_size]
        with torch.no_grad():
            emb = model.encode(
                batch,
                batch_size=len(batch),
                show_progress_bar=False,
                convert_to_numpy=True,
            )
        embeddings.append(emb.astype("float32"))

    return np.concatenate(embeddings, axis=0)


def build_and_save_embeddings(
    model_name: str,
    test_mode: bool,
    config: dict[str, Any],
) -> None:
    validate_model_name(model_name, config["available_models"])

    cte_path, embeddings_dir = resolve_paths(config)
    if not cte_path.exists():
        raise FileNotFoundError(
            f"CTE.csv not found at {cte_path}. "
            "Make sure the file exists or set cte_csv_path in config/ml_config.yaml."
        )

    print(f"ML root: {ML_ROOT}")
    print(f"Loading CTE data from: {cte_path}")

    df = pd.read_csv(cte_path)
    if test_mode:
        df = df.head(100).copy()
        print(f"Test mode enabled: using first {len(df)} rows.")
    else:
        print(f"Full mode: using all {len(df)} rows.")

    if "CTE_id" not in df.columns:
        raise KeyError("Column 'CTE_id' not found in CTE.csv.")

    df["text"] = df.apply(build_item_text, axis=1)

    texts = df["text"].tolist()
    cte_ids = df["CTE_id"].to_numpy()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Encoding {len(texts)} items with model: {model_name}")

    embeddings = encode_texts(model_name, texts, device=device)

    embeddings_dir.mkdir(parents=True, exist_ok=True)

    model_short = model_name.replace("/", "_")
    prefix = "test_" if test_mode else ""

    emb_path = embeddings_dir / f"{prefix}{model_short}.npy"
    ids_path = embeddings_dir / f"{prefix}cte_ids.npy"

    np.save(emb_path, embeddings)
    np.save(ids_path, cte_ids)

    print(f"Saved embeddings to: {emb_path}")
    print(f"Saved CTE ids to: {ids_path}")


def main() -> None:
    try:
        config = load_config()
    except FileNotFoundError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    args = parse_args(config)
    try:
        build_and_save_embeddings(
            model_name=args.model,
            test_mode=args.test,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

