import os
import yaml
import faiss
import torch
from typing import Any, List
from sentence_transformers import SentenceTransformer

def validate_model_name(model_name: str, available_models: List[str]) -> None:
    if model_name not in available_models:
        msg = (
            f"Unsupported model '{model_name}'. "
            f"Supported models are: {', '.join(available_models)}"
        )
        raise ValueError(msg)

class SearchSystem():
    def __init__(self, model_name, index_path) -> None:
        validate_model_name(model_name)
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu" 
            self.model = SentenceTransformer(
                model_name,
                trust_remote_code=True,
                device=device
            )
            print(f"Initialized model: {model_name} on {device}")
            
            self.index = faiss.read_index(index_path)
            print(f"Initialized index {index_path}")
            
        except Exception as e:
            print(f"ERROR in initializing SearchSystem: {e}")

    def search(query:str, top_k:int):
        ...