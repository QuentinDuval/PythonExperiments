import copy
import itertools

import torch
import torch.nn as nn
import torch.optim as optim

from Learning.Dataset import *
from Learning.Evaluation import *
from Learning.Prediction import *
from Learning.Ratio import *


class Predictor:
    def __init__(self, model: nn.Module, vectorizer: Vectorizer, with_gradient_clipping=False, split_seed=None):
        self.model = model
        self.vectorizer = vectorizer
        self.with_gradient_clipping = with_gradient_clipping
        self.split_seed = split_seed
        self.batch_size = 100
        self.loss_function = nn.CrossEntropyLoss()
        self.min_epoch = 20
        self.max_epoch_no_progress = 30
        self.data_augmentation = lambda x: None

    def fit(self, training_corpus: CommitMessageCorpus, learning_rate=1e-3, weight_decay=0):
        data_set = CommitMessageDataSet.from_corpus(corpus=training_corpus, vectorizer=self.vectorizer)
        return self.fit_dataset(data_set, learning_rate, weight_decay)

    def fit_dataset(self, data_set: Dataset, learning_rate=1e-3, weight_decay=0):
        training_set, validation_set = data_set.split(0.9, seed=self.split_seed)
        self.data_augmentation(training_set)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size, shuffle=True)

        best_model = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        max_valid_accuracy = Ratio()

        for epoch in itertools.count(0):
            if epoch - best_epoch > min(max(self.min_epoch, epoch // 5), self.max_epoch_no_progress):
                break

            train_accuracy = self._training_pass(training_loader, optimizer)
            valid_accuracy = self._validation_pass(validation_loader)

            if valid_accuracy > max_valid_accuracy:
                best_model = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                max_valid_accuracy = valid_accuracy

            if epoch % 5 == 0:
                print("-" * 50)
                print("Epoch:", epoch)
                print("-" * 50)
                print("Training:", train_accuracy)
                print("Validation:", valid_accuracy)

        self.model.load_state_dict(best_model)
        print("Training (max):", self._validation_pass(training_loader))
        print("Validation (max):", max_valid_accuracy)
        return max_valid_accuracy.to_percentage()

    def _training_pass(self, training_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer):
        self.model.train()
        train_accuracy = Ratio()
        for minibatch in training_loader:
            inputs, labels = minibatch['x'], minibatch['y']
            outputs = self.model(inputs)
            loss = self.loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            if self.with_gradient_clipping:
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            optimizer.step()
            train_accuracy += self._compute_accuracy(outputs, labels)
        return train_accuracy

    def _validation_pass(self, validation_loader: torch.utils.data.DataLoader):
        self.model.eval()
        valid_accuracy = Ratio()
        for minibatch in validation_loader:
            inputs, labels = minibatch['x'], minibatch['y']
            outputs = self.model(inputs)
            valid_accuracy += self._compute_accuracy(outputs, labels)
        return valid_accuracy

    def _compute_accuracy(self, outputs, labels):
        _, predicted = torch.max(outputs.data, dim=-1)
        predicted = predicted.view(-1)  # Useful in case there are more than 1 dimension left (ex: sequence prediction)
        labels = labels.view(-1)        # Useful in case there are more than 1 dimension left (ex: sequence prediction)
        return Ratio((predicted == labels).sum().item(), len(labels))

    def predict(self, sentence):
        x = torch.FloatTensor(self.vectorizer.vectorize(sentence=sentence))
        x = x.unsqueeze(dim=0)
        self.model.eval()
        y = self.model(x)
        confidence, predicted = torch.max(y.data, dim=-1)
        target_label = CommitMessageCorpus.target_class_label(predicted.item())
        return Prediction(target_label, confidence.item())

    def evaluate(self, test_corpus: CommitMessageCorpus):
        all_expected = []
        all_predicted = []
        test_set = CommitMessageDataSet.from_corpus(corpus=test_corpus, vectorizer=self.vectorizer)

        self.model.eval()
        for minibatch in torch.utils.data.DataLoader(test_set, batch_size=self.batch_size, shuffle=False):
            inputs, labels = minibatch['x'], minibatch['y']
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_expected.extend(labels)
            all_predicted.extend(predicted)

        ConfusionMatrix(all_expected, all_predicted, CommitMessageCorpus.TARGET_CLASSES).show()
