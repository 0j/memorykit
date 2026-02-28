"""
embedder.py: Converts text into vector embeddings.
Supports OpenAI (cloud) and sentence-transformers (local/free).
"""

from typing import List
import os


class Embedder:
    """
    Wraps embedding models. Defaults to local (free) model,
    falls back to OpenAI if API key is set.
    """

    def __init__(self, model: str = "auto"):
        self.model_name = model
        self._client = None
        self._local_model = None
        self._mode = self._detect_mode(model)
        print(f"✓ Embedder initialized in '{self._mode}' mode")

    def _detect_mode(self, model: str) -> str:
        if model == "auto":
            if os.getenv("OPENAI_API_KEY"):
                return "openai"
            else:
                return "local"
        return model

    def _load_openai(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()

    def _load_local(self):
        if self._local_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._local_model = SentenceTransformer("all-MiniLM-L6-v2")
                print("✓ Local embedding model loaded (free, no API key needed)")
            except ImportError:
                raise ImportError(
                    "Please install sentence-transformers: pip install sentence-transformers\n"
                    "Or set OPENAI_API_KEY to use OpenAI embeddings."
                )

    def embed(self, text: str) -> List[float]:
        """Convert a string to a vector embedding."""
        if self._mode == "openai":
            return self._embed_openai(text)
        else:
            return self._embed_local(text)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple texts at once (more efficient)."""
        if self._mode == "openai":
            return [self._embed_openai(t) for t in texts]
        else:
            self._load_local()
            return self._local_model.encode(texts).tolist()

    def _embed_openai(self, text: str) -> List[float]:
        self._load_openai()
        response = self._client.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def _embed_local(self, text: str) -> List[float]:
        self._load_local()
        return self._local_model.encode([text])[0].tolist()
