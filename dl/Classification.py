import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plot

from dl.SplitDataset import *


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

            print("Training:", training_loss)
            print("Validation:", validation_loss)

    def predict(self, x):
        x = torch.FloatTensor([x])
        y = self.model(x)
        _, predicted = torch.max(y, 1)
        return predicted.item()


def show_result(points, classif):
    blue_x = [points[i][0] for i in range(len(classif)) if classif[i] == 1]
    blue_y = [points[i][1] for i in range(len(classif)) if classif[i] == 1]
    green_x = [points[i][0] for i in range(len(classif)) if classif[i] == 0]
    green_y = [points[i][1] for i in range(len(classif)) if classif[i] == 0]

    plot.scatter(blue_x, blue_y, c='b')
    plot.scatter(green_x, green_y, c='g')
    plot.show()


def test_classif_product_positive():
    points = []
    classif = []
    for _ in range(1000):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        points.append(np.array([x, y], dtype=np.float32))
        classif.append(1 if x * y > 0 else 0)
    points = np.stack(points)
    classif = np.stack(classif)

    model = MultilayerClassifier(input_size=2, hidden_size=10, output_size=2)
    predictor = ClassificationPredictor(model=model)
    predictor.fit(data_set=SplitDataset(points, classif), epoch=100, learning_rate=0.1)

    '''
    Show the results :)
    '''

    # show_result(points, classif)

    predicted = []
    for p in points:
        predicted.append(predictor.predict(p))
    show_result(points, predicted)


test_classif_product_positive()



