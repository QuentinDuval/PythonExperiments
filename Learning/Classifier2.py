import torch.nn.functional as fn

from Learning.Classifier1 import *
from Learning.Predictor import *


class RegexVectorizer(Vectorizer):
    """
    The terms are selected by expertise and also statistical correlation
    The model tries to find correct weights for these terms, to best predict the class of the commit
    """

    def __init__(self):
        self.features = HardCodedClassifier.features_list +\
                        HardCodedClassifier.fixes_list +\
                        HardCodedClassifier.refactoring_list +\
                        HardCodedClassifier.revert_list

    def vectorize(self, fix_description):
        fix_description = fix_description.lower()
        vector = [1 if feat in fix_description else 0 for feat in self.features]
        return torch.FloatTensor(vector)

    def get_vocabulary_len(self):
        return len(self.features)


class PerceptronModel(nn.Module):
    def __init__(self, vocabulary_len, nb_classes):
        super().__init__()
        self.vocabulary_len = vocabulary_len
        self.nb_classes = nb_classes
        self.output_layer = nn.Linear(self.vocabulary_len, self.nb_classes)
        torch.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x, apply_softmax=True):
        x = self.output_layer(x)
        if apply_softmax:
            x = fn.softmax(x, dim=-1)
        return x


class DoublePerceptronModel(nn.Module):
    def __init__(self, vocabulary_len, hidden_dimension, nb_classes, drop_out=0.0):
        super().__init__()
        self.vocabulary_len = vocabulary_len
        self.hidden_dimension = hidden_dimension
        self.nb_classes = nb_classes
        self.drop_out_p = drop_out
        self.input_layer = nn.Linear(self.vocabulary_len, self.hidden_dimension)
        self.output_layer = nn.Linear(self.hidden_dimension, self.nb_classes)
        torch.nn.init.xavier_normal_(self.input_layer.weight)
        torch.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x, apply_softmax=True):
        x = self.input_layer(x)
        x = self.drop_out(x)
        x = self.output_layer(fn.relu(x))
        if apply_softmax:
            x = fn.softmax(x, dim=-1)
        return x

    def drop_out(self, x):
        if self.drop_out_p > 0:
            return fn.dropout(x, p=self.drop_out_p, training=self.training)
        return x


class TriplePerceptronModel(nn.Module):
    def __init__(self, vocabulary_len, hidden_dimension, nb_classes, drop_out=0.0):
        super().__init__()
        self.vocabulary_len = vocabulary_len
        self.hidden_dimension = hidden_dimension
        self.nb_classes = nb_classes
        self.drop_out_p = drop_out
        self.input_layer = nn.Linear(self.vocabulary_len, self.hidden_dimension)
        # self.input_layer_bn = nn.BatchNorm1d(self.hidden_dimension)
        self.middle_layer = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        # self.middle_layer_bn = nn.BatchNorm1d(self.hidden_dimension)
        self.output_layer = nn.Linear(self.hidden_dimension, self.nb_classes)
        # torch.nn.init.xavier_normal_(self.input_layer.weight)
        # torch.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x, apply_softmax=True):
        x = self.input_layer(x)
        # x = self.input_layer_bn(x)
        x = self.drop_out(x)
        x = self.middle_layer(fn.relu(x))
        # x = self.middle_layer_bn(x)
        x = self.drop_out(x)
        x = self.output_layer(fn.relu(x))
        if apply_softmax:
            x = fn.softmax(x, dim=-1)
        return x

    def drop_out(self, x):
        if self.drop_out_p > 0:
            return fn.dropout(x, p=self.drop_out_p, training=self.training)
        return x


"""
Testing...
"""


def test_model_2(split_seed=None):
    vectorizer = RegexVectorizer()
    vocab_len = vectorizer.get_vocabulary_len()

    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')

    model = PerceptronModel(vocabulary_len=vocab_len, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus)
    predictor.evaluate(test_corpus=test_corpus)

    print("-" * 50)

    model = DoublePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=3)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus)
    predictor.evaluate(test_corpus=test_corpus)

    print("-" * 50)

    model = TriplePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=3)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus)
    predictor.evaluate(test_corpus=test_corpus)


"""
Observe that:
1) by multiplying the number of layers we gain a bit of precision
2) the model generalizes pretty good (simple models tend to do so) 

Model 1:
----------------------------------------
Training (max): 2899/4244 (68.30819981149858%)
Validation (max): 311/472 (65.88983050847457%)
------------------------------
Accuracy: 62.297496318114874 %

Model 2:
----------------------------------------
Training (max): 3178/4244 (74.88218661639962%)
Validation (max): 327/472 (69.27966101694916%)
------------------------------
Accuracy: 63.770250368188506 %

Model 3:
------------------------------
Training (max): 3216/4244 (75.7775683317625%)
Validation (max): 338/472 (71.61016949152543%)
------------------------------
Accuracy: 63.91752577319587 %
"""

# test_model_2(split_seed=0)
