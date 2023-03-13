from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward
import torch
from torch import nn, Tensor


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dimension,
        number_of_heads,
        feed_forward_intermediate_dimension,
        attention_dropout_p,
        feed_forward_dropout_p,
    ) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(
            attention_dropout_p=attention_dropout_p,
            number_of_heads=number_of_heads,
            keys_embedding_dimension=embedding_dimension,
            queries_embedding_dimension=embedding_dimension,
            values_embedding_dimension=embedding_dimension,
            queries_and_keys_projection_dimension=embedding_dimension,
            values_projection_dimension=embedding_dimension,
        )
        self.feed_forward = PositionWiseFeedForward(
            embedding_dimension,
            feed_forward_intermediate_dimension,
            feed_forward_dropout_p,
        )

        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)

    def forward(self, embeddings, mask):
        query = key = value = self.layer_norm_1(embeddings)
        embeddings = self.attention(query, key, value, mask) + embeddings
        embeddings = self.feed_forward(self.layer_norm_2(embeddings)) + embeddings
        return embeddings


def get_shifted_right_attention_mask(sequence_length: int) -> torch.Tensor:
    return torch.ones((sequence_length, sequence_length), dtype=torch.bool).tril_(
        diagonal=0
    )


class CausalLayer(EncoderLayer):
    def forward(self, embeddings: Tensor):
        batch_size, sequence_length, embedding_dimension = embeddings.shape
        mask = get_shifted_right_attention_mask(sequence_length).to(embeddings.device)
        return super().forward(embeddings, mask=mask)


class DecoderLayer(nn.Module):
    def __init__(
        self,
        encoder_embedding_dimension,
        decoder_embedding_dimension,
        number_of_heads,
        feed_forward_intermediate_dimension,
        attention_dropout_p,
        feed_forward_dropout_p,
    ) -> None:
        super().__init__()
        self.attention_1 = MultiHeadAttention(
            attention_dropout_p=attention_dropout_p,
            number_of_heads=number_of_heads,
            keys_embedding_dimension=decoder_embedding_dimension,
            queries_embedding_dimension=decoder_embedding_dimension,
            values_embedding_dimension=decoder_embedding_dimension,
            queries_and_keys_projection_dimension=decoder_embedding_dimension,
            values_projection_dimension=decoder_embedding_dimension,
        )

        self.attention_2 = MultiHeadAttention(
            attention_dropout_p=attention_dropout_p,
            number_of_heads=number_of_heads,
            queries_embedding_dimension=decoder_embedding_dimension,
            keys_embedding_dimension=encoder_embedding_dimension,
            values_embedding_dimension=encoder_embedding_dimension,
            queries_and_keys_projection_dimension=decoder_embedding_dimension,
            values_projection_dimension=decoder_embedding_dimension,
        )

        self.feed_forward = PositionWiseFeedForward(
            decoder_embedding_dimension,
            feed_forward_intermediate_dimension,
            feed_forward_dropout_p,
        )

        self.layer_norm_1 = nn.LayerNorm(decoder_embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(decoder_embedding_dimension)
        self.layer_norm_3 = nn.LayerNorm(decoder_embedding_dimension)

    def forward(self, embeddings, mask, encoder_output):
        query = key = value = self.layer_norm_1(embeddings)
        embeddings = self.attention_1(query, key, value, mask) + embeddings

        query = self.layer_norm_2(embeddings)
        key = value = encoder_output
        embeddings = self.attention_2(query, key, value, mask) + embeddings

        embeddings = self.feed_forward(self.layer_norm_3(embeddings)) + embeddings
        return embeddings


class LearnablePositionalEncoder(nn.Module):
    def __init__(self, max_sequence_length: int, embedding_dimension: int) -> None:
        super().__init__()
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.embedding = nn.Embedding(max_sequence_length, embedding_dimension)

    def forward(self, embeddings: Tensor) -> Tensor:
        batch_size, sequence_length, embedding_dimension = embeddings.shape
        positions = torch.arange(
            sequence_length, dtype=torch.long, device=embeddings.device
        ).unsqueeze(0)
        return self.embedding(positions) + embeddings


class TokenEmbedder(nn.Embedding):
    def __init__(self, vocabulary_size, embedding_dimension) -> None:
        super().__init__(vocabulary_size, embedding_dimension)
