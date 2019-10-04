import copy
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.utils.data import DataLoader


class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return fn.log_softmax(x, dim=-1)


class TwoLayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )
        # init.xavier_normal_()

    def forward(self, x):
        x = self.fc(x)
        return fn.log_softmax(x, dim=-1)


class ClassificationPredictor:
    def __init__(self, model: nn.Module):
        self.model = model

    def fit(self, data_set: SplitDataset, epoch: int, learning_rate: float, weight_decay: float = 0):
        loss_fct = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        training_set, validation_set = data_set.split(ratio=0.9)
        training_loader = DataLoader(training_set, batch_size=100, shuffle=True)
        validation_loader = DataLoader(validation_set, batch_size=100, shuffle=False)

        best_model = copy.deepcopy(self.model)
        best_validation = float('inf')

        for epoch in range(epoch):

            training_loss = 0
            self.model.train()
            for x, target in training_loader:
                self.model.zero_grad()
                outputs = self.model(x)
                loss = loss_fct(outputs, target)
                loss.backward()
                optimizer.step()
                training_loss += loss.item()

            validation_loss = 0
            self.model.eval()
            for x, target in validation_loader:
                outputs = self.model(x)
                loss = loss_fct(outputs, target)
                validation_loss += loss.item()

            if validation_loss < best_validation:
                best_validation = validation_loss
                best_model = copy.deepcopy(self.model)

            print("Training:", training_loss)
            print("Validation:", validation_loss)

        self.model = best_model

    def predict(self, x):
        x = torch.FloatTensor([x])
        y = self.model(x)
        _, predicted = torch.max(y, 1)
        return predicted.item()


def sample_points(classifier, x_bounds=(-1,1), y_bounds=(-1,1), count=1000):
    points = []
    expected = []
    for _ in range(count):
        x = np.random.uniform(x_bounds[0], x_bounds[1])
        y = np.random.uniform(y_bounds[0], y_bounds[1])
        points.append(np.array([x, y], dtype=np.float32))
        expected.append(classifier(x, y))
    return np.stack(points), np.array(expected, dtype=np.int64)


def test_classif_product_positive(model):
    classif = lambda x, y: 1 if x * y > 0 else 0

    points, expected = sample_points(classif, x_bounds=(-1, 1), y_bounds=(-1, 1), count=2000)
    predictor = ClassificationPredictor(model=model)
    predictor.fit(data_set=SplitDataset(points, expected), epoch=100, learning_rate=0.1)

    points, expected = sample_points(classif, x_bounds=(-1, 1), y_bounds=(-1, 1), count=2000)
    predicted = [predictor.predict(p) for p in points]
    show_result(points, expected, predicted)


def test_classif_above_x2(model, noise=0.):
    classif = lambda x, y: 1 if y > x * x else 0
    def with_noise(classif):
        def wrapped(x, y):
            return classif(x + np.random.normal(0, noise, size=1), y + np.random.normal(0, noise, size=1))
        return wrapped

    points, expected = sample_points(with_noise(classif), x_bounds=(-2, 2), y_bounds=(-1, 3), count=2000)
    predictor = ClassificationPredictor(model=model)
    predictor.fit(data_set=SplitDataset(points, expected), epoch=100, learning_rate=0.1)

    points, expected = sample_points(classif, x_bounds=(-2, 2), y_bounds=(-1, 3), count=2000)
    predicted = [predictor.predict(p) for p in points]
    show_result(points, expected, predicted)


def test_classif_circle(model):
    classif = lambda x, y: 1 if x ** 2 + y ** 2 <= 1 else 0

    points, expected = sample_points(classif, x_bounds=(-2, 2), y_bounds=(-2, 2), count=2000)
    predictor = ClassificationPredictor(model=model)
    predictor.fit(data_set=SplitDataset(points, expected), epoch=100, learning_rate=0.1)

    points, expected = sample_points(classif, x_bounds=(-2, 2), y_bounds=(-2, 2), count=2000)
    predicted = [predictor.predict(p) for p in points]
    show_result(points, expected, predicted)


def test_classif_circle_border(model, nb_classes=2, polar=False):
    def classif(x, y):
        norm = x ** 2 + y ** 2
        if norm <= 0.7:
            return 0 % nb_classes
        if norm <= 1.7:
            return 1 % nb_classes
        return 2 % nb_classes

    def to_polar(point):
        r = math.sqrt(point[0] ** 2 + point[1] ** 2)
        t = math.acos(point[0] / r)
        return np.array([r, t], dtype=np.float32)

    def to_polars(points):
        points = [to_polar(p) for p in points]
        return np.stack(points)

    points, expected = sample_points(classif, x_bounds=(-2, 2), y_bounds=(-2, 2), count=2000)
    if polar:
        points = to_polars(points)
    predictor = ClassificationPredictor(model=model)
    predictor.fit(data_set=SplitDataset(points, expected), epoch=200, learning_rate=1e-3)

    points, expected = sample_points(classif, x_bounds=(-2, 2), y_bounds=(-2, 2), count=2000)
    predicted = [predictor.predict(p if not polar else to_polar(p)) for p in points]
    show_result(points, expected, predicted)


# test_classif_product_positive(model=TwoLayerClassifier(input_size=2, hidden_size=10, output_size=2))
# test_classif_product_positive(model=LinearClassifier(input_size=2, output_size=2))

# test_classif_above_x2(model=TwoLayerClassifier(input_size=2, hidden_size=10, output_size=2))
# test_classif_above_x2(model=TwoLayerClassifier(input_size=2, hidden_size=10, output_size=2), noise=0.3)
# test_classif_above_x2(model=LinearClassifier(input_size=2, output_size=2))

# test_classif_circle(model=TwoLayerClassifier(input_size=2, hidden_size=10, output_size=2))
# test_classif_circle(model=LinearClassifier(input_size=2, output_size=2))

"""
This example is really interesting.

Recognizing the border is much harder than finding the circle with only 2 classes.
- It seems much harder to optimize: the initial condition makes everything (stuck in local minima or not)

Whereas, 3 classes makes it easy to optimize, and the solution is found easily.
"""
# test_classif_circle_border(nb_classes=2, model=TwoLayerClassifier(input_size=2, hidden_size=10, output_size=2))
# test_classif_circle_border(nb_classes=2, polar=True, model=TwoLayerClassifier(input_size=2, hidden_size=10, output_size=2))
# test_classif_circle_border(nb_classes=3, model=TwoLayerClassifier(input_size=2, hidden_size=25, output_size=3))
# test_classif_circle_border(nb_classes=2, model=LinearClassifier(input_size=2, output_size=2))
# test_classif_circle_border(nb_classes=2, polar=True, model=LinearClassifier(input_size=2, output_size=2))


