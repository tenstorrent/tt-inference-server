class EmbeddingResponse:
    def __init__(self, embedding: list[float], total_tokens: int):
        self.embedding = embedding
        self.total_tokens = total_tokens
