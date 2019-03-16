from Learning.Corpus import *
from Learning.Evaluation import *


class HardCodedClassifier:
    """
    Manually hardcoded classifier:
    - Uses a list of terms to search, and a series of 'if statements' to predict the type of the commit
    - Serves as baseline for all our future evaluations

    The terms are selected by expertise and also statistical correlation
    """

    refactoring_list =\
        ["util", "extract", "refact", "rework", "minor", "clean", "raii", "mef", "technical", "function",
        "move", "renam", "simplif", "split", "dead", "code", "enabler", "use", "miss", "call", "private",
        "fitness", "cucumber", "decoupl", "modular", "get", "set", "layout", "sonar"]
    revert_list = ["revert"]
    fixes_list =\
        ["fix", "solve", "correct", "rather", "possibility", "crash", "bau", "check", "slo"]
    features_list =\
        ["feature", "improve", "enhance", "modif", "add", "to", "allow", "now", "require", "should", "must",
        "as", "log", "audit", "user", "client", "view", "display", "configur", "perform", "optim", "improve",
        "increas", "latency", "caching", "speed", "throughput", "bulk", "index", "group", "sav", "load", "store"]

    def __init__(self):
        print("Feature len:", len(self.refactoring_list + self.revert_list + self.fixes_list + self.features_list))

    def predict(self, fix_description):
        fix_description = fix_description.lower()
        for token in self.refactoring_list:
            if token in fix_description:
                return 0
        for token in self.features_list:
            if token in fix_description:
                return 1
        for token in self.revert_list:
            if token in fix_description:
                return 2
        for token in self.fixes_list:
            if token in fix_description:
                return 3
        return 1

    def evaluate(self):
        corpus = CommitMessageCorpus.from_split('test')
        predicted = []
        expected = []
        for x, y in corpus:
            predicted.append(self.predict(x))
            expected.append(corpus.target_class_index(y))
        print_confusion_matrix(expected, predicted, CommitMessageCorpus.TARGET_CLASSES)


def test_model_1():
    predictor = HardCodedClassifier()
    predictor.evaluate()


# test_model_1()
