from Learning.Classifier4 import *
from Learning.Predictor import *


class ConvolutionalModel(nn.Module):
    def __init__(self, vocabulary_len, embedding_size, sentence_size, nb_classes, drop_out=0.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.sentence_size = sentence_size
        self.nb_classes = nb_classes
        self.drop_out_p = drop_out

        self.embed = nn.Embedding(vocabulary_len, self.embedding_size)
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_size, out_channels=self.embedding_size, kernel_size=5, padding=2),
            nn.ELU()
        )
        self.output_layer = nn.Linear(self.embedding_size * self.sentence_size * 2, self.nb_classes)

        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x):
        batch_size, sequence_len = x.shape
        x = self.embed(x)                   # Return (batch size, sequence size, embedding size)

        conv = x.permute(0, 2, 1)           # Returns (batch size, embedding size, sequence size)
        conv = self.convnet(conv)           # Returns (batch size, feature size, sequence size)
        conv = conv.view((batch_size, -1))  # Returns (batch size, feature size * sequence size)

        x = x.view((batch_size, -1))        # Returns (batch size, embedding size * sequence size)
        x = torch.cat((x, conv), dim=1)     # Returns (batch size, (embedding size + feature size) * sequence size)

        x = self.drop_out(x)
        x = self.output_layer(x)
        return fn.softmax(x, dim=-1)

    def drop_out(self, x):
        if self.drop_out_p > 0:
            return fn.dropout(x, p=self.drop_out_p, training=self.training)
        return x


def test_model_5(split_seed=None):
    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')

    vectorizer = EmbeddingVectorizer.from_corpus(training_corpus, NltkTokenizer(), max_length=30)
    vocab_len = vectorizer.get_vocabulary_len()
    sentence_len = vectorizer.max_length

    model = ConvolutionalModel(vocabulary_len=vocab_len, sentence_size=sentence_len,
                               embedding_size=20, nb_classes=3, drop_out=0.5)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-3, weight_decay=1e-4)
    predictor.evaluate(test_corpus=test_corpus)


"""
Training (max): 3864/4221 (91.54228855721394%)
Validation (max): 373/469 (79.53091684434968%)
--------------------------------------------------
Accuracy: 73.40267459138187 %
"""

# test_model_5(split_seed=0)
