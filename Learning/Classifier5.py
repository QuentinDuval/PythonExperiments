from Learning.Classifier4 import *
from Learning.Predictor import *


class ConvolutionalModel(nn.Module):
    def __init__(self, vocabulary_len, embedding_size, sentence_size, nb_classes):
        super().__init__()
        self.embedding_size = embedding_size
        self.sentence_size = sentence_size
        self.nb_classes = nb_classes
        self.embed = nn.Embedding(vocabulary_len, self.embedding_size)

        self.convnet = nn.Sequential(
            nn.Conv1d(in_channels=self.embedding_size, out_channels=self.embedding_size * 2, kernel_size=5),
            nn.ELU(),
            nn.Conv1d(in_channels=self.embedding_size * 2, out_channels=self.embedding_size, kernel_size=5),
            nn.ELU()
        )

        '''
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size,
                               out_channels=self.embedding_size,
                               kernel_size=5, padding=2, stride=1)
        self.conv_relu = nn.ELU()
        self.max_pool1 = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        '''

        # self.drop_out = nn.Dropout(0.5)
        self.output_layer = nn.Linear(self.embedding_size * (self.sentence_size - 8), self.nb_classes)

        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x):
        batch_size = x.size()[0]
        x = self.embed(x)           # Return (batch size, sequence size, embedding size)
        x = x.permute(0, 2, 1)      # Returns (batch size, embedding size, sequence size)

        '''
        x = self.conv1(x)
        x = self.conv_relu(x)
        x = self.max_pool1(x)
        '''

        x = self.convnet(x)
        # x = self.drop_out(x)

        x = x.view((batch_size, -1))

        x = self.output_layer(x)
        return fn.softmax(x, dim=-1)


def test_model_5(split_seed=None):
    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')

    vectorizer = EmbeddingVectorizer.from_corpus(training_corpus, NltkTokenizer(), max_length=30)
    vocab_len = vectorizer.get_vocabulary_len()
    sentence_len = vectorizer.max_length

    model = ConvolutionalModel(vocabulary_len=vocab_len, sentence_size=sentence_len, embedding_size=10, nb_classes=3)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=1e-4, weight_decay=1e-4)
    predictor.evaluate(test_corpus=test_corpus)


"""
Training (max): 3612/4244 (85.10838831291234%)
Validation (max): 338/472 (71.61016949152543%)
------------------------------
Accuracy: 71.5758468335788 %
"""

# test_model_5(split_seed=0)
