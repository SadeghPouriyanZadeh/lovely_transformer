from .attention import MultiHeadAttention
from .feedforward import PositionWiseFeedForward
import torch
from torch import nn


class EncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dimension,
        number_of_heads,
        feed_forward_intermediate_dimnesion,
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
            feed_forward_intermediate_dimnesion,
            feed_forward_dropout_p,
        )

        self.layer_norm_1 = nn.LayerNorm(embedding_dimension)
        self.layer_norm_2 = nn.LayerNorm(embedding_dimension)

    def forward(self, embeddings, mask):
        query = key = value = self.layer_norm_1(embeddings)
        embeddings = self.attention(query, key, value, mask) + embeddings
        embeddings = self.feed_forward(self.layer_norm_2(embeddings)) + embeddings
        return embeddings


class DecoderLayer(nn.Module):
    def __init__(
        self,
        encoder_embedding_dimension,
        decoder_embedding_dimension,
        number_of_heads,
        feed_forward_intermediate_dimnesion,
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
            feed_forward_intermediate_dimnesion,
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
