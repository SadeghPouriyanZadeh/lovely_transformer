import torch
from torch import nn
from typing import Optional
from torch import Tensor


class ScaledDotProductAttention(nn.Module):
    """scaled dot product attention

    Args:
        dropout_p (float, optional): Dropout probability. Defaults to 0.0
    """

    def __init__(self, dropout_p: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        """performed scaled dot-product attention

        Note about shapes:
            - B: Batch Size
            - S1: Number of queries
            - S2: Number of keys-value pairs
            - E1: Embedding dimention of queries and keys
            - E2: Embedding dimention of values

        Args:
            query (torch.Tensor): (B, S1, E1)
            key (torch.Tensor): (B, S2, E1)
            value (torch.Tensor): (B, S2, E2)
            mask (Optional[torch.Tensor], optional): . Defaults to None.

        Returns:
            Tensor: (B, S1, E2)
        """
        dim_k = key.shape[2]
        scores = torch.bmm(query, key.transpose(1, 2)) / (dim_k) ** 0.5
        if mask is not None:
            scores = scores.masked_fill(mask == False, float("-inf"))
        weights = torch.softmax(scores, dim=2)
        weights = self.dropout(weights)
        attention = torch.bmm(weights, value)
        return attention


class AttentionHead(nn.Module):
    def __init__(
        self,
        queries_embedding_dimension: int,
        keys_embedding_dimension: int,
        values_embedding_dimension: int,
        queries_and_keys_projection_dimension: int,
        values_projection_dimension: int,
        attention_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.queries_embedding_dimension = queries_embedding_dimension
        self.keys_embedding_dimension = keys_embedding_dimension
        self.values_embedding_dimension = values_embedding_dimension
        self.queries_and_keys_projection_dimension = (
            queries_and_keys_projection_dimension
        )
        self.values_projection_dimension = values_projection_dimension

        self.queries_projection = nn.Linear(
            self.queries_embedding_dimension,
            self.queries_and_keys_embedding_dimension,
        )
        self.keys_projection = nn.Linear(
            self.keys_embedding_dimension,
            self.queries_and_keys_embedding_dimension,
        )
        self.scaled_dot_product_attention = ScaledDotProductAttention(
            attention_dropout_p
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ):
        """performed scaled dot-product attention

        Note about shapes:
            - B: Batch Size
            - S1: Number of queries
            - S2: Number of keys-value pairs
            - E1: Embedding dimention of queries
            - E2: Embedding dimention of keys
            - E3: Embedding dimention of values
            - E4: Projection dimention of values

        Args:
            query (torch.Tensor): (B, S1, E1)
            key (torch.Tensor): (B, S2, E2)
            value (torch.Tensor): (B, S2, E3)
            mask (Optional[torch.Tensor], optional): . Defaults to None.

        Returns:
            Tensor: (B, S1, E4)
        """
        projected_queries = self.queries_projection(query)
        projected_keys = self.queries_projection(key)
        projected_values = self.queries_projection(value)
        attention = self.scaled_dot_product_attention(
            projected_queries, projected_keys, projected_values, mask
        )
        return attention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        number_of_heads: int,
        queries_embedding_dimension: int,
        keys_embedding_dimension: int,
        values_embedding_dimension: int,
        queries_and_keys_projection_dimension: int,
        values_projection_dimension: int,
        attention_dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.number_of_heads = number_of_heads
        self.queries_embedding_dimension = queries_embedding_dimension
        self.keys_embedding_dimension = keys_embedding_dimension
        self.values_embedding_dimension = values_embedding_dimension
        self.queries_and_keys_projection_dimension = (
            queries_and_keys_projection_dimension
        )
        self.values_projection_dimension = values_projection_dimension
        self.attention_dropout_p = attention_dropout_p

        self.head_queries_and_keys_projection_dimension = (
            queries_and_keys_projection_dimension // number_of_heads
        )
        self.head_values_projection_dimension = (
            values_projection_dimension // number_of_heads
        )
        self.heads = nn.ModuleList()
        for _ in range(self.number_of_heads):
            self.heads.append(
                AttentionHead(
                    self.queries_embedding_dimension,
                    self.keys_embedding_dimension,
                    self.values_embedding_dimension,
                    self.head_queries_and_keys_projection_dimension,
                    self.head_values_projection_dimension,
                    self.attention_dropout_p,
                )
            )
        self.output_projection = nn.Linear(
            self.number_of_heads @ self.head_values_projection_dimension,
            self.values_projection_dimension,
        )

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None,
    ):
        head_outputs = [h(query, key, value, mask) for h in self.heads]
        concatenated_outputs = torch.cat(head_outputs, dim=2)


class SelfAttentionHead(AttentionHead):
    def __init__(self, embedding_dimension, attention_dropout_p: float = 0) -> None:
        super().__init__(
            embedding_dimension,
            embedding_dimension,
            embedding_dimension,
            embedding_dimension,
            embedding_dimension,
            attention_dropout_p,
        )

    def forward(self, embeddings: Tensor):
        """performs scaled dot product attention

        Note about shapes:
            - B: Batch Size
            - S: Number of queries
            - E: Embedding dimension

        Args:
            embeddings (Tensor): (B, S, E)
        """
        super().forward(embeddings, embeddings, embeddings)
