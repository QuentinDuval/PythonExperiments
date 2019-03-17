from sklearn import metrics
import tabulate


def print_percentage(title, value):
    print(title, value * 100, "%")


class ConfusionMatrix:
    def __init__(self, expected, predicted, target_classes):
        self.accuracy = metrics.accuracy_score(expected, predicted)
        self.expected = expected
        self.predicted = predicted
        self.target_classes = target_classes

    def show(self):
        print("-" * 50)
        print_percentage("Accuracy:", self.accuracy)
        print()
        print("Classification report: ")
        print(metrics.classification_report(self.expected, self.predicted, target_names=self.target_classes))
        print("Confusion matrix: ")
        print(tabulate.tabulate(metrics.confusion_matrix(self.expected, self.predicted), self.target_classes))
