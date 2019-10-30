
import numpy as np

import torch
from torch.nn.parameter import Parameter

from ..utils import is_categorical
from ..init import initialize
from ..plot import scatter_reduce


DEFAULT_INITIALIZER = dict(
    name="truncated_normal",
    params=dict(
        std=0.01
    )
)


class EmbeddingEncoder(torch.nn.Embedding):

    def from_df(cls, df, column, embedding_size=None, **kwargs):
        """Create and EmbeddingEncoder using a dataframe category column.

        **kwargs: any parameter of the EmbeddingEncoder constructor other than
            the labels. The labels are taken from the category, if you want
            to specify your own labels use the default constructor.
        """

        assert is_categorical(df, column)
        num_embeddings = len(df[column].cat.categories)
        labels = df[column].cat.categories
        return EmbeddingEncoder(num_embeddings, embedding_size=embedding_size,
                                labels=labels, **kwargs)

    def __init__(self, num_embeddings, embedding_size=None, labels=None,
                 initializer=None, dropout=None, **kwargs):
        """An enhanced version of `torch.nn.Embedding`.

        num_embeddings (int): cardinality of the categorical value that you
            want to embed.
        embedding_size (int): dimension of the embedding.
            If not size is specify we assing an embedding size using the
            expression: `min(ceil(sqrt(num_embeddings)), 50)`.
            The authors of the entity embeddings paper
            (https://arxiv.org/abs/1604.06737) use: `num_embeddings - 1`
            as a default dimension of the embeddings.
        labels (list): Name of the categories. The length of the list should
            be `num_embeddings`.
        dropout (`None` or a float in the interval `(0, 1)`): Indicates the
            quantity of dropout to apply to the embeddings.
        initializer (str or dict): Initializer function used in the embedding
            weights.
        """

        if embedding_size is None:
            embedding_size = min(int(np.ceil(np.sqrt(num_embeddings))), 50)
        super().__init__(num_embeddings, embedding_size)

        if initializer is None:
            initializer = DEFAULT_INITIALIZER
        if labels is None:
            # Use index as labels if not labels are defined.
            labels = np.arange(num_embeddings)
        if bool(dropout):
            dropout = torch.nn.Dropout(dropout)
            self.add_module("embedding_dropout", dropout)
        # Save properties.
        self.set_labels(labels)
        self.dropout = dropout
        self.initializer = initializer
        # Init weights.
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            initialize(self.weight, self.initializer)

    def forward(self, x):
        emb = super().forward(x)
        if self.dropout is not None:
            return self.dropout(emb)
        return emb

    def set_labels(self, labels):
        assert len(labels) == self.num_embeddings
        self.labels = labels

    def get_weights(self):
        return self.weight.data

    def set_weights(self, w):
        """Update weights of encoding layer.

        It also updates the number of embeddings and the embedding size.

        FIXME: this way to update the weights is probably not correct,
            but it works for inference. Be careful if you want to train
            the network after this, maybe the learner (adam, sgd...)
            needs to be updated with the new parameters.
        """
        self.num_embeddings = w.shape[0]
        self.embedding_dim = w.shape[1]
        self.weight = Parameter(w)

    def save_weights(self, filename):
        """Save the embedding weights to a file using torch. """

        torch.save(self.weight, filename)

    def get_gensim_model(self):
        """Return a gensim model initialized with the current weights.

        You can use gensim to query the embedding space: find most similar
        embeddings using cosine distance, perform embedding arithmetic
        (like the very famouse "king - man + woman = queen"), finding the
        intruder embedding...

        raise: If gensim is not installed, raise an exception.
        """

        from gensim.models import KeyedVectors

        model = KeyedVectors(self.embedding_dim)
        model.add(self.labels, self.get_weights().cpu().numpy())
        return model

    def get_output_length(self):
        return self.embedding_dim

    @property
    def shape(self):
        return [self.num_embeddings, self.embedding_dim]

    def plot(self, hoverinfo="text", **kwargs):
        w = self.get_weights().cpu().numpy()
        text = self.labels
        return scatter_reduce(w, text=text, hoverinfo=hoverinfo, **kwargs)

