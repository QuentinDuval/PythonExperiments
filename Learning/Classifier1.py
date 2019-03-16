from Learning.Corpus import *
from Learning.Evaluation import *


class HardCodedClassifier:
    """
    Manually hardcoded classifier
    - Uses a list of terms to search, and a series of 'if statements' to predict the type of the commit
    - Serves as baseline for all our future evaluations
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

    def predict(self, fix_description):
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

    @classmethod
    def evaluate(cls):
        print("Feature len:", len(cls.refactoring_list + cls.revert_list + cls.fixes_list + cls.features_list))

        corpus = CommitMessageCorpus.from_split('test')
        model = cls()
        predicted = []
        expected = []
        for x, y in corpus:
            predicted.append(model.predict(x))
            expected.append(corpus.target_class_index(y))

        # TODO - investigate why it is so weird for the fixes (because it is at the end...)
        print_confusion_matrix(expected, predicted, CommitMessageCorpus.TARGET_CLASSES)


def test_model_1():
    HardCodedClassifier.evaluate()


# test_model_1()
