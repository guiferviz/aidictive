
from torch import nn
from torch.nn.parameter import Parameter


class EmbeddingEncoder(nn.Embedding, Encoder):

    def __init__(self, df, metadata_dict, labels=None):
        metadata_dict = EMBEDDING_SCHEMA.validate(metadata_dict)
        num_embeddings = len(df[metadata_dict["name"]].cat.categories)
        embedding_size = metadata_dict["embedding_size"]
        if embedding_size is None:
            embedding_size = int(np.ceil(np.sqrt(num_embeddings)))
        super().__init__(num_embeddings, embedding_size)
        init_range = metadata_dict["embedding_init_range"]
        if init_range is not None:
            self.weight.data.uniform_(-init_range, init_range)
        if labels is None:
            labels = df[metadata_dict["name"]].cat.categories
        dropout = metadata_dict["embedding_dropout"]
        if dropout is not None:
            dropout = nn.Dropout(dropout)
            self.add_module("embedding_dropout", dropout)
        self.set_labels(labels)
        self.dropout = dropout
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            trunc_normal_(self.weight, std=0.01)

    def forward(self, x):
        if self.dropout is not None:
            return self.dropout(super().forward(x))
        return super().forward(x)

    def set_labels(self, labels):
        assert len(labels) == self.num_embeddings
        self.labels = labels

    def get_weights(self):
        return self.weight.data

    def set_weights(self, w):
        """Update weights of encoding layer.

        It also updates the number of embeddings and the embedding size.

        WARNING: this way to update the weights is probably not correct,
            but it works for inference. Be careful if you want to train
            the network after this, maybe the learner (adam, sgd...)
            needs to be updated with the new parameters.
        """
        self.num_embeddings = w.shape[0]
        self.embedding_dim = w.shape[1]
        self.weight = Parameter(w)

    def save_weights(self, filename):
        torch.save(self.weight, filename)

    def get_labels(self):
        return self.labels

    def get_gensim_model(self):
        from gensim.models import KeyedVectors
        model = KeyedVectors(self.embedding_dim)
        model.add(self.labels, self.get_weights().cpu().numpy())
        return model

    def get_output_length(self):
        return self.embedding_dim

    @property
    def shape(self):
        return [self.num_embeddings, self.embedding_dim]


def trunc_normal_(tensor, mean=0.0, std=1.0):
    """Truncated normal.

    Taken from:
    https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/12
    """
    size = tensor.shape
    tmp = tensor.new_empty(size + (4,)).normal_()
    valid = (tmp < 2) & (tmp > -2)
    ind = valid.max(-1, keepdim=True)[1]
    tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
    tensor.data.mul_(std).add_(mean)
    return tensor
