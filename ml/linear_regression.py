"""
Linear regression:
- estimate a single scalar (dependent value) from a list of explanatory variables
- with MSE (Mean Squared Error) has a closed form.

Do not mix with:
- Multivariate regression: linear regression but with multiple dependent values (TODO - use DL? parameter sharing)
- Logistic regression (estimate a boolean with a probability) which does not have a closed form
  => has to use SGD or Newton's method to solve it


Problem
-------
y = f(x) of the form:
* y = x . w + b
* y = x^t w + b

Just add a 1 at the end of x:
* y = x^t w
* with one more weight to learn

Learning
--------

You have several examples (one line for each example - with 1 at the end for the intercept)
y = X w

Example (to show with a nice scatter plot):
(3, 4)
(5, 4)
(7, 6)
(9, 8)
(11, 8)

Matrices:

X =
[[3 1]
 [5 1]
 [7 1]
 [9 1]
 [11 1]]

y =
[[4],
 [4],
 [6],
 [8],
 [8]]

And we have to find w^t = [slope intercept]


Demonstration of the closed form for MSE
--------------------------------

We want to find argmin(w) for norm(y - X w)

norm (y - X w) = (y - X w) ^ t (y - X w) = y^t y + w^t X^t X w - 2 w^t X^t y (you can transpose a scalar)

To find the minimum, solve for the gradient equal to 0.
The gradient of this norm is equal to: 2 X^t X w - 2 X^t y

=> Solve: X^t X w = X^t y
=> Complexity is O(D ^ 3) with gaussian elimination + O(N * D ^ 2) for the matrix products

TODO - compare with implementation in terms of matrix inversion (compute time)
TODO - compare with DL when you have lots of inputs and lots of dimensions (generate inputs with gaussian noise)


Solving techniques
------------------
* Inverting the matrix is costly (only to be used when you need to multiply by the reverse several times)
* Gaussian elimination is the best way when you have to do it just once
"""

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


"""
Implementation based on the closed formula for the MSE error
"""


def linear_regression(xs, ys):
    X = np.stack(np.array(x, dtype=np.float32) for x in xs)
    X = np.hstack((X, np.ones(shape=(len(xs), 1))))
    y = np.array(ys)
    XT = np.transpose(X)
    try:
        w = np.linalg.solve(XT @ X, XT @ y)
        return w
    except np.linalg.LinAlgError as e:
        print(e)
        return None


class LinearRegression:
    """
    Packaging the model to be more easily used
    """
    def __init__(self, regression_method):
        self.regression_method = regression_method
        self.slope = 0
        self.intercept = 0

    def fit(self, xs, ys):
        self.slope, self.intercept = self.regression_method(xs, ys)

    def transform(self, xs):
        return self.intercept + np.array(xs) * self.slope

    @property
    def parameters(self):
        return {'slope': self.slope, 'intercept': self.intercept}


"""
Packaging the model to include feature engineering:
The transformation of input allows to accommodate for more models (like quadratic, cubic, etc)
"""


class MappedRegression:
    def __init__(self, regression_method, transformation):
        self.regression_method = regression_method
        self.transformation = transformation
        self.weights = self.transformation([(0.,)])
        self.intercept = 0

    def fit(self, xs, ys):
        ws = self.regression_method(self.transformation(xs), ys)
        self.weights = ws[:-1]
        self.intercept = ws[-1]

    def transform(self, xs):
        return self.intercept + self.transformation(xs) @ self.weights

    @property
    def parameters(self):
        return {'weights': self.weights, 'intercept': self.intercept}


"""
Implementation based on PyTorch and usual deep learning practices
"""


class Progress:
    def __init__(self, limit):
        self.increasing = 0
        self.limit = limit
        self.previous = float('inf')

    def new_loss(self, val):
        if val >= self.previous:
            self.increasing += 1
        else:
            self.increasing = 0
        self.previous = val

    def __bool__(self):
        return self.increasing < self.limit


def regression_dl(xs, ys, model):
    inputs = torch.tensor(xs, dtype=torch.float32, requires_grad=False)
    expected = torch.tensor(ys, dtype=torch.float32, requires_grad=False)
    training_loader = DataLoader(TensorDataset(inputs, expected), batch_size=100, shuffle=True)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.)
    criterion = nn.MSELoss()

    # TODO - implement the derivative stuff by hand

    progress = Progress(limit=20)
    while progress:
        total_loss = 0.
        for xs, ys in training_loader:
            optimizer.zero_grad()
            output = model(xs)
            loss = criterion(output.squeeze(dim=1), ys)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        progress.new_loss(total_loss)

    model.eval()
    return model


def linear_regression_dl(xs, ys):
    feature_size = len(xs[0])
    model = regression_dl(xs, ys, model=nn.Linear(feature_size, 1))
    weights = model.weight.detach().reshape((feature_size,)).numpy()
    bias = model.bias.detach().numpy()
    return np.hstack((weights, bias))


"""
Implementation based on multiple layers
"""


class MultiLayerRegressionDL:
    """
    Packaging the model to be more easily used
    """
    def __init__(self, regression_method):
        self.regression_method = regression_method
        self.model = None

    def fit(self, xs, ys):
        self.model = self.regression_method(xs, ys)

    def transform(self, xs):
        xs = torch.tensor(xs, dtype=torch.float32, requires_grad=False)
        return self.model(xs).detach().numpy()

    @property
    def parameters(self):
        return {}

