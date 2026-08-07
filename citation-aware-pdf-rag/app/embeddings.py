"""Vector embedding generation module using SentenceTransformers."""

from typing import Sequence

from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around SentenceTransformer for generating vector embeddings locally."""

    DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME) -> None:
        """Initialize the EmbeddingModel with a SentenceTransformer model.

        Args:
            model_name: The HuggingFace repository name or local path of the model.

        Raises:
            ValueError: If model_name is empty or not a string.
        """
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string.")

        self.model_name = model_name
        self._model = SentenceTransformer(model_name)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        """Return the vector embedding dimension of the loaded model.

        Returns:
            The embedding vector size (e.g., 384 for all-MiniLM-L6-v2).
        """
        return self._dimension

    def embed_text(self, text: str) -> list[float]:
        """Generate a vector embedding for a single text.

        Args:
            text: A single input string.

        Returns:
            A list of floats representing the embedding vector.

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"Input text must be a string, got {type(text).__name__}")

        embedding = self._model.encode(text, convert_to_numpy=True)
        return [float(val) for val in embedding.tolist()]

    def embed_batch(self, texts: Sequence[str]) -> list[list[float]]:
        """Generate vector embeddings for a sequence of texts.

        Args:
            texts: A sequence of input strings.

        Returns:
            A list of vector embeddings (lists of floats), preserving input ordering.

        Raises:
            TypeError: If texts is a single string or if any element is not a string.
        """
        if isinstance(texts, str):
            raise TypeError("embed_batch expects a sequence of strings (e.g. list), not a single string.")

        if not isinstance(texts, (list, tuple, Sequence)):
            raise TypeError(f"Input texts must be a sequence of strings, got {type(texts).__name__}")

        if not texts:
            return []

        for i, item in enumerate(texts):
            if not isinstance(item, str):
                raise TypeError(f"Item at index {i} must be a string, got {type(item).__name__}")

        embeddings = self._model.encode(list(texts), convert_to_numpy=True)
        return [[float(val) for val in vec] for vec in embeddings.tolist()]
