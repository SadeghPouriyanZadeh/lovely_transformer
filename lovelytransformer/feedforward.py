import torch
from torch import nn
from typing import Optional
from torch import Tensor


class PositionWiseFeedForward(nn.Module):
    def __init__(
        self, embedding_dimension: int, intermediate_dimension: int, dropout_p: float
    ) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.intermediate_dimension = intermediate_dimension
        self.dropout_p = dropout_p

        self.layers = nn.Sequential(
            nn.Linear(self.embedding_dimension, self.intermediate_dimension),
            nn.GELU(),
            nn.Linear(self.embedding_dimension, self.intermediate_dimension),
            nn.Dropout(self.dropout_p),
        )

    def forward(self, embeddings: Tensor) -> Tensor:
        return self.layers(embeddings)
