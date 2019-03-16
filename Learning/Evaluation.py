from sklearn import metrics
import tabulate


def print_percentage(title, value):
    print(title, value * 100, "%")


def print_confusion_matrix(expected, predicted, target_classes):
    print("-" * 30)
    print_percentage("Accuracy:", metrics.accuracy_score(expected, predicted))
    print()

    print("Classification report: ")
    print(metrics.classification_report(expected, predicted, target_names=target_classes))

    print("Confusion matrix: ")
    print(tabulate.tabulate(metrics.confusion_matrix(expected, predicted), target_classes))

    # Print accuracy / recall
    # matrix = metrics.confusion_matrix(expected, predicted)
    # print(matrix.diagonal() / matrix.sum(axis=0))
    # print(matrix.diagonal() / matrix.sum(axis=1))

