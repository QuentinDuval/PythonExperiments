from Learning.Classifier4 import *
from Learning.Predictor import *


"""
Convolutional model
"""


class ConvolutionalModel(nn.Module):
    def __init__(self, vocabulary_len, embedding_size, sentence_size, nb_classes, drop_out=0.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.sentence_size = sentence_size
        self.nb_classes = nb_classes
        self.drop_out_p = drop_out
        self.conv_channels = self.embedding_size

        self.embed = nn.Embedding(vocabulary_len, self.embedding_size)
        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_size, out_channels=self.conv_channels, kernel_size=5, padding=2),
            nn.ELU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.output_layer = nn.Linear(self.embedding_size * self.sentence_size + self._conv_out(), self.nb_classes)

        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.output_layer.weight)

    def _conv_out(self):
        x = torch.zeros((1, self.embedding_size, self.sentence_size))
        y = self.convnet(x)
        _, feature_size, sequence_size = y.shape
        return feature_size * sequence_size

    def forward(self, x, apply_softmax=True):
        x = x.long()
        batch_size, sequence_len = x.shape
        x = self.embed(x)                   # Return (batch size, sequence size, embedding size)

        conv = x.permute(0, 2, 1)           # Returns (batch size, embedding size, sequence size)
        conv = self.convnet(conv)           # Returns (batch size, feature size, sequence size)
        conv = conv.view((batch_size, -1))  # Returns (batch size, feature size * sequence size)

        x = x.view((batch_size, -1))        # Returns (batch size, embedding size * sequence size)
        x = torch.cat((x, conv), dim=1)     # Returns (batch size, (embedding size + feature size) * sequence size)

        x = self.drop_out(x)
        x = self.output_layer(x)
        if apply_softmax:
            x = fn.softmax(x, dim=-1)
        return x

    def drop_out(self, x):
        if self.drop_out_p > 0:
            return fn.dropout(x, p=self.drop_out_p, training=self.training)
        return x


"""
Packaging the model
"""


"""
Packaging the model
"""


class CommitClassifier5:
    def __init__(self, model, vocabulary: Vocabulary, max_sentence_length):
        self.model = model
        self.vocabulary = vocabulary
        self.max_sentence_length = max_sentence_length
        self.tokenizer = NltkTokenizer()
        self.vectorizer = EmbeddingVectorizer(vocabulary=vocabulary,
                                              tokenizer=self.tokenizer,
                                              max_length=self.max_sentence_length)
        self.predictor = Predictor(model=model, vectorizer=self.vectorizer)
        self.predictor.with_gradient_clipping = True

    def predict(self, sentence: str):
        return self.predictor.predict(sentence)

    def save(self, file_name):
        dump = {
            'model': self.model,
            'vocabulary': self.vocabulary,
            'max_sentence_length': self.max_sentence_length
        }
        torch.save(dump, file_name)

    @classmethod
    def load(cls, file_name):
        dump = torch.load(file_name)
        classifier = cls(model=dump['model'],
                         vocabulary=dump['vocabulary'],
                         max_sentence_length=dump['max_sentence_length'])
        return classifier


def load_best_model():
    classifier = CommitClassifier5.load('models/convolutional.model')
    return classifier.predictor


"""
Testing...
"""


def test_model_5(split_seed=None):
    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')
    sentence_len = 30

    vocabulary = Vocabulary.from_corpus(training_corpus, tokenizer=NltkTokenizer(), min_freq=2, add_unknowns=True)
    model = ConvolutionalModel(vocabulary_len=len(vocabulary),
                               sentence_size=sentence_len,
                               embedding_size=15, nb_classes=3, drop_out=0.5)

    classifier = CommitClassifier5(model=model, vocabulary=vocabulary, max_sentence_length=sentence_len)
    classifier.predictor.split_seed = split_seed
    classifier.predictor.fit(training_corpus=training_corpus, learning_rate=1e-3, weight_decay=3e-4, learning_rate_decay=0.98)
    classifier.predictor.evaluate(test_corpus=test_corpus)

    for commit in ['quantity was wrong',
                   'move CollateralAgreement to lib/folder1/folder2',
                   'add tab in collateral screen to show statistics',
                   'use smart pointers to simplify memory management of ClassName']:
        print(commit)
        print(classifier.predict(commit))
    answer = input('save model (Y/N)? >')
    if answer.lower() == "y":
        print("saving model...")
        classifier.save('models/convolutional.model')


"""
Training (max): 3926/4222 (92.98910468972052%)
Validation (max): 377/470 (80.2127659574468%)
--------------------------------------------------
Accuracy: 77.26597325408619 %

quantity was wrong
fix (90.91%)
move CollateralAgreement to lib/folder1/folder2
refactor (95.96%)
add tab in collateral screen to show statistics
feat (90.87%)
use smart pointers to simplify memory management of ClassName
refactor (90.46%)
"""

# test_model_5(split_seed=0)
