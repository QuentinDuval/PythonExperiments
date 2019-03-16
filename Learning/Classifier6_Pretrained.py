from Learning.Classifier6 import *
from Learning.Predictor import *
from Learning.WordEmbeddings import *


class PretrainedRnnClassifier(nn.Module):
    def __init__(self, pretrained_embeddings: WordEmbeddings, nb_classes):
        super().__init__()
        self.embed = pretrained_embeddings.embedding
        self.hidden_size = 15
        self.rnn = nn.GRU(input_size=pretrained_embeddings.embedding_size,
                          hidden_size=self.hidden_size,
                          num_layers=1,
                          batch_first=False,
                          dropout=0)
        self.output_layer = nn.Linear(self.hidden_size, nb_classes)
        nn.init.xavier_normal_(self.embed.weight)
        nn.init.xavier_normal_(self.output_layer.weight)

    def parameters(self):
        yield from self.rnn.parameters()
        yield from self.output_layer.parameters()

    def forward(self, x, x_length=None):
        batch_size, sequence_len = x.shape

        x = self.embed(x)       # Shape is batch_size, sequence_len, embedding_size
        x = x.permute(1, 0, 2)  # Shape is sequence_len, batch_size, embedding_size

        init_state = torch.zeros(1, batch_size, self.hidden_size)
        outputs, final_state = self.rnn(x, init_state)

        # outputs shape: sequence_len, batch_size, hidden_size
        # final state shape: 1, batch_size, hidden_size

        x = final_state.squeeze(0)
        # x = outputs[-1]
        # x = self.select_column(outputs, x_length)

        # x = self.drop_out(x)
        x = self.output_layer(x)
        return fn.log_softmax(x, dim=-1)

    def select_column(self, rnn_out, x_length):
        x_length = x_length.long().detach().numpy() - 1
        out = []
        for batch_index, column_index in enumerate(x_length):
            out.append(rnn_out[column_index, batch_index])
        return torch.stack(out)


def test_model_6_pretrained(split_seed=None):
    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')
    pretrained_embeddings = WordEmbeddings.load_from(model_path='resources/unsupervised_model.bin')
    vectorizer = EmbeddingRnnVectorizer(pretrained_embeddings.get_vocabulary(), NltkTokenizer(), max_length=50)

    model = PretrainedRnnClassifier(pretrained_embeddings=pretrained_embeddings, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus, learning_rate=0.01, weight_decay=0.0)
    predictor.evaluate(test_corpus=test_corpus)


"""
Training (max): 2730/4244 (64.32610744580585%)
Validation (max): 305/472 (64.61864406779661%)
------------------------------
Accuracy: 63.03387334315169 %
"""

# TODO - improve this shit...

# test_model_6_pretrained(split_seed=0)

