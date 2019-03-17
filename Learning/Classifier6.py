import numpy as np
import torch.nn.functional as fn

from Learning.Predictor import *
from Learning.Tokenizer import *
from Learning.Vocabulary import *


class EmbeddingRnnVectorizer:
    def __init__(self, vocabulary: Vocabulary, tokenizer, max_length=None):
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self.max_length = max_length

    def vectorize(self, sentence):
        tokens = [self.vocabulary.START] + self.tokenizer(sentence) + [self.vocabulary.END]
        tokens = [self.vocabulary.word_lookup(token) for token in tokens]
        max_length = self.max_length or len(tokens)
        features = np.zeros(max_length, dtype=np.int64)
        if max_length >= len(tokens):
            features[:len(tokens)] = tokens
            features[len(tokens):] = self.vocabulary.word_lookup(self.vocabulary.PADDING)
        else:
            features[:max_length] = tokens[:max_length]
        return features

    def get_vocabulary_len(self):
        return len(self.vocabulary)

    @classmethod
    def from_corpus(cls, corpus: CommitMessageCorpus, tokenizer, min_freq, max_length):
        vocabulary = Vocabulary.from_corpus(corpus=corpus, tokenizer=tokenizer, min_freq=min_freq, add_unknowns=True)
        return cls(vocabulary=vocabulary, tokenizer=tokenizer, max_length=max_length)


class RnnClassifier(nn.Module):
    def __init__(self, vocabulary_len, embedding_size, hidden_size, nb_classes, drop_out):
        super().__init__()
        self.embed = nn.Embedding(vocabulary_len, embedding_size)
        self.hidden_size = hidden_size
        self.drop_out_p = drop_out
        self.rnn = nn.GRU(input_size=embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=False,
                          dropout=0)
        self.output_layer = nn.Linear(self.hidden_size, nb_classes)
        # nn.init.xavier_normal_(self.embed.weight)
        # nn.init.xavier_normal_(self.rnn.all_weights)
        # nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x):
        x = x.long()
        batch_size, sequence_len = x.shape
        x = self.embed(x)       # Shape is batch_size, sequence_len, embedding_size
        x = x.permute(1, 0, 2)  # Shape is sequence_len, batch_size, embedding_size

        init_state = torch.zeros(1, batch_size, self.hidden_size)
        outputs, final_state = self.rnn(x, init_state)
        x = final_state.squeeze(0)  # shape was: 1, batch_size, hidden_size

        x = self.drop_out(x)
        x = self.output_layer(x)
        return fn.softmax(x, dim=-1)

    def drop_out(self, x):
        if self.drop_out_p > 0:
            return fn.dropout(x, p=self.drop_out_p, training=self.training)
        return x


def test_model_6(split_seed=None):
    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')

    vectorizer = EmbeddingRnnVectorizer.from_corpus(training_corpus, NltkTokenizer(), min_freq=2, max_length=50)
    vocab_len = vectorizer.get_vocabulary_len()

    model = RnnClassifier(vocabulary_len=vocab_len, embedding_size=20, hidden_size=20, nb_classes=3, drop_out=0.5)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-2, weight_decay=1e-4)
    predictor.evaluate(test_corpus=test_corpus)
    return predictor


def test_model_6_interactive():
    predictor = test_model_6()
    while True:
        sentence = input("> ")
        print(predictor.predict(sentence))


"""
embedding_size=20, hidden_size=20, nb_classes=3, drop_out=0.5
learning_rate=1e-2, weight_decay=1e-4
------------------------------
Training (max): 3576/4221 (84.71926083866383%)
Validation (max): 362/470 (77.02127659574468%)
------------------------------
Accuracy: 74.33234421364985 %
"""

# test_model_6(split_seed=0)
# test_model_6_interactive()

