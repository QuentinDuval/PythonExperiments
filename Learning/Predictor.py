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
        self.split_ratio = 0.9
        self.split_seed = split_seed
        self.batch_size = 100
        self.loss_function = nn.CrossEntropyLoss()
        self.min_epoch = 20
        self.max_epoch = None
        self.max_epoch_no_progress = 30
        self.data_augmentation = lambda x: None

    def fit(self, training_corpus: CommitMessageCorpus, learning_rate=1e-3, weight_decay=0):
        data_set = CommitMessageDataSet.from_corpus(corpus=training_corpus, vectorizer=self.vectorizer)
        return self.fit_dataset(data_set, learning_rate, weight_decay)

    def fit_dataset(self, data_set: CommitMessageDataSet, learning_rate=1e-3, weight_decay=0):
        training_set, validation_set = data_set.split(self.split_ratio, seed=self.split_seed)
        self.data_augmentation(training_set)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        training_loader = torch.utils.data.DataLoader(training_set, batch_size=self.batch_size, shuffle=True)
        validation_loader = torch.utils.data.DataLoader(validation_set, batch_size=self.batch_size, shuffle=True)

        best_model = copy.deepcopy(self.model.state_dict())
        best_epoch = 0
        max_valid_accuracy = Ratio()

        for epoch in itertools.count(0):
            if self.max_epoch and epoch > self.max_epoch:
                break
            if epoch - best_epoch > min(max(self.min_epoch, epoch // 5), self.max_epoch_no_progress):
                break

            train_loss, train_accuracy = self._training_pass(training_loader, optimizer)
            valid_accuracy = self._validation_pass(validation_loader)

            if valid_accuracy > max_valid_accuracy:
                best_model = copy.deepcopy(self.model.state_dict())
                best_epoch = epoch
                max_valid_accuracy = valid_accuracy

            if epoch % 5 == 0:
                print("-" * 50)
                print("Epoch:", epoch)
                print("-" * 50)
                print("Training loss:", train_loss)
                print("Training:", train_accuracy)
                print("Validation:", valid_accuracy)

        self.model.load_state_dict(best_model)
        print("Training (max):", self._validation_pass(training_loader))
        print("Validation (max):", max_valid_accuracy)
        return max_valid_accuracy.to_percentage()

    def _training_pass(self, training_loader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer):
        self.model.train()
        train_accuracy = Ratio()
        train_loss = 0.0
        for minibatch in training_loader:
            inputs, labels = minibatch['x'], minibatch['y']
            optimizer.zero_grad()
            outputs = self.model(inputs, apply_softmax=False)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            if self.with_gradient_clipping:
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.25)
            optimizer.step()
            train_loss += loss.item()
            train_accuracy += self._compute_accuracy(outputs, labels)
        return train_loss, train_accuracy

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
        # TODO - inject a vocabulary to translate the element back to realm of text instead of hard-coding it

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
