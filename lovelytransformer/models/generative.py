from torch import nn, Tensor
import torch

from ..buildingblocks.layers import (
    CausalLayer,
    TokenEmbedder,
    LearnablePositionalEncoder,
)


class GenerativeTransformer(nn.Module):
    def __init__(
        self,
        vocabulary_size,
        embedding_dimension,
        max_sequence_length,
        number_of_layers,
        number_of_heads,
        feed_forward_intermediate_dimension,
        attention_dropout_p,
        feed_forward_dropout_p,
    ) -> None:
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.token_embedder = TokenEmbedder(
            vocabulary_size=vocabulary_size, embedding_dimension=embedding_dimension
        )
        self.positional_encoder = LearnablePositionalEncoder(
            embedding_dimension=embedding_dimension,
            max_sequence_length=max_sequence_length,
        )
        self.causal_layers = nn.Sequential()

        for _ in range(number_of_layers):
            self.causal_layers.append(
                CausalLayer(
                    embedding_dimension,
                    number_of_heads,
                    feed_forward_intermediate_dimension,
                    attention_dropout_p,
                    feed_forward_dropout_p,
                )
            )
        self.linear_classifier = nn.Linear(
            in_features=embedding_dimension, out_features=vocabulary_size
        )

    def forward(self, tokens):
        embeddings = self.token_embedder(tokens)
        embeddings = self.positional_encoder(embeddings)
        embeddings = self.causal_layers(embeddings)
        logits = self.linear_classifier(embeddings)
        return logits


def get_gpt_model(vocabulary_size, max_sequence_length, scale=1):
    embedding_dimension = round(768 * scale)
    number_of_layers = round(12 * scale)
    number_of_heads = round(12 * scale)
    feed_forward_intermediate_dimension = round(3072 * scale)
    attention_dropout_p = 0
    feed_forward_dropout_p = 0
    return GenerativeTransformer(
        vocabulary_size=vocabulary_size,
        max_sequence_length=max_sequence_length,
        embedding_dimension=embedding_dimension,
        number_of_layers=number_of_layers,
        number_of_heads=number_of_heads,
        feed_forward_intermediate_dimension=feed_forward_intermediate_dimension,
        attention_dropout_p=attention_dropout_p,
        feed_forward_dropout_p=feed_forward_dropout_p,
    )
