import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.utils.data import DataLoader

from dl.ClassificationVisualization import *
from dl.SplitDataset import *


class LinearClassifier(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x):
        x = self.fc(x)
        return fn.softmax(x, dim=-1)


class MultilayerClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        x = self.fc(x)
        return fn.softmax(x, dim=-1)


class ClassificationPredictor:
    def __init__(self, model: nn.Module):
        self.model = model

    def fit(self, data_set: SplitDataset, epoch: int, learning_rate: float):
        loss_fct = nn.CrossEntropyLoss()
        optimizer = optim.Adam(params=self.model.parameters(), lr=learning_rate)
        training_set, validation_set = data_set.split(ratio=0.9)
        training_loader = DataLoader(training_set, batch_size=100, shuffle=True)
        validation_loader = DataLoader(validation_set, batch_size=100, shuffle=False)

        best_model = copy.deepcopy(self.model)
        best_validation = float('inf')

        for epoch in range(epoch):

            training_loss = 0
            self.model.train()
            for x, target in training_loader:
                outputs = self.model(x)
                loss = loss_fct(outputs, target)
                optimizer.zero_grad()
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
    return np.stack(points), np.stack(expected)


def test_classif_product_positive(model):
    points, expected = sample_points(lambda x, y: 1 if x * y > 0 else 0,
                                     x_bounds=(-1, 1),
                                     y_bounds=(-1, 1),
                                     count=1000)

    predictor = ClassificationPredictor(model=model)
    predictor.fit(data_set=SplitDataset(points, expected), epoch=100, learning_rate=0.1)

    predicted = [predictor.predict(p) for p in points]
    show_result(points, expected, predicted)


def test_classif_two_x2(model):
    points, expected = sample_points(lambda x, y: 1 if y > x ** 2 else 0,
                                     x_bounds=(-2, 2),
                                     y_bounds=(-1, 3),
                                     count=1000)

    predictor = ClassificationPredictor(model=model)
    predictor.fit(data_set=SplitDataset(points, expected), epoch=100, learning_rate=0.1)

    predicted = [predictor.predict(p) for p in points]
    show_result(points, expected, predicted)


# test_classif_product_positive(model=MultilayerClassifier(input_size=2, hidden_size=10, output_size=2))
# test_classif_product_positive(model=LinearClassifier(input_size=2, output_size=2))
# test_classif_two_x2(model=MultilayerClassifier(input_size=2, hidden_size=10, output_size=2))
# test_classif_two_x2(model=LinearClassifier(input_size=2, output_size=2))



