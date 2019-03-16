import torch.nn.functional as fn

from Learning.Predictor import *


class Vectorizer2(Vectorizer):
    """
    Extract a given set of words and tag with 1 or 0 whether this word is there or not
    """

    def __init__(self):
        self.features = []

        # Terms related to refactoring
        self.features += ["util", "extract", "refact", "rework", "minor", "clean", "raii", "mef", "technical",
                          "function", "move", "renam", "simplif", "split", "dead", "code", "enabler", "use", "miss",
                          "call", "private", "fitness", "cucumber", "decoupl", "modular", "get", "set", "layout", "sonar"]

        # Terms related to fixes
        self.features += ["fix", "solve", "correct", "rather", "possibility", "crash", "bau", "check", "mlk"]

        # Terms related to features
        self.features += ["feature", "improve", "enhance", "modif", "add", "to", "allow", "now", "require", "should",
                          "must", "as", "log", "audit", "user", "client", "view", "display", "configur"]

        # Terms related to revert
        self.features += ["revert"]

        # Terms related to performance
        self.features += ["perform", "optim", "improve", "increas", "latency", "caching", "speed", "throughput",
                          "bulk", "index", "group", "sav", "load", "store"]

        # General terms
        self.features += ["algorithm", "class", "api", "depend", "warn", "change", "handle"]

    def vectorize(self, fix_description):
        data_point = []
        fix_description = fix_description.lower()
        for feat in self.features:
            data_point.append(1 if feat in fix_description else 0)
        return torch.LongTensor(data_point)

    def get_vocabulary_len(self):
        return len(self.features)


class PerceptronModel(nn.Module):
    def __init__(self, vocabulary_len, nb_classes):
        super().__init__()
        self.input_dimension = vocabulary_len
        self.nb_classes = nb_classes
        self.output_layer = nn.Linear(self.input_dimension, self.nb_classes)
        torch.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x):
        x = self.output_layer(x.float())
        return fn.softmax(x, dim=-1)


class MultiPerceptronModel(nn.Module):
    def __init__(self, vocabulary_len, hidden_dimension, nb_classes):
        super().__init__()
        self.input_dimension = vocabulary_len
        self.hidden_dimension = hidden_dimension
        self.nb_classes = nb_classes
        self.input_layer = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.output_layer = nn.Linear(self.hidden_dimension, self.nb_classes)
        torch.nn.init.xavier_normal_(self.input_layer.weight)
        torch.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x):
        x = self.input_layer(x.float())
        x = fn.relu(x)
        x = self.output_layer(x)
        return fn.softmax(x, dim=-1)


class TriplePerceptronModel(nn.Module):
    def __init__(self, vocabulary_len, hidden_dimension, nb_classes):
        super().__init__()
        self.input_dimension = vocabulary_len
        self.hidden_dimension = hidden_dimension
        self.nb_classes = nb_classes
        self.input_layer = nn.Linear(self.input_dimension, self.hidden_dimension)
        self.middle_layer = nn.Linear(self.hidden_dimension, self.hidden_dimension)
        self.output_layer = nn.Linear(self.hidden_dimension, self.nb_classes)
        torch.nn.init.xavier_normal_(self.input_layer.weight)
        torch.nn.init.xavier_normal_(self.output_layer.weight)

    def forward(self, x):
        x = self.input_layer(x.float())
        x = fn.relu(x)
        x = self.middle_layer(x)
        x = fn.relu(x)
        x = self.output_layer(x)
        return fn.softmax(x, dim=-1)


"""
Testing...
"""


def test_model_2(split_seed=None):
    vectorizer = Vectorizer2()
    vocab_len = vectorizer.get_vocabulary_len()

    training_corpus = CommitMessageCorpus.from_split('train')
    test_corpus = CommitMessageCorpus.from_split('test')

    model = PerceptronModel(vocabulary_len=vocab_len, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus)
    predictor.evaluate(test_corpus=test_corpus)

    print("-" * 50)

    model = MultiPerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=4)
    predictor = Predictor(model=model, vectorizer=vectorizer, with_gradient_clipping=True, split_seed=split_seed)
    predictor.fit(training_corpus=training_corpus)
    predictor.evaluate(test_corpus=test_corpus)

    print("-" * 50)

    model = TriplePerceptronModel(vocabulary_len=vocab_len, hidden_dimension=100, nb_classes=4)
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
