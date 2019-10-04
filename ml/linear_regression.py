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


def linear_regression_dl(xs, ys):
    criterion = nn.MSELoss()
    model = nn.Linear(len(xs[0]), 1)
    model.train()

    optimizer = optim.Adam(model.parameters(), lr=1e-2, weight_decay=0.)
    inputs = torch.stack([torch.FloatTensor(x) for x in xs])
    training_set = TensorDataset(inputs, torch.FloatTensor(ys))
    training_loader = DataLoader(training_set, batch_size=100, shuffle=True)

    # TODO - continue while the loss decreases (stop when it does not improve)
    # TODO - check how we can get rid of the reshape
    # TODO - implement the derivative stuff by hand

    for _ in range(10_000):
        total_loss = 0.
        for xs, ys in training_loader:
            optimizer.zero_grad()
            output = model(xs)
            loss = criterion(output.reshape((5,)), ys)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(total_loss)

    model.eval()
    return model.weight, model.bias     # TODO - return the weights, not tensors


# Non singular matrix are solved correctly
ws = linear_regression(xs=[(3,), (5,), (7,), (9,), (11,)], ys=[4, 4, 6, 8, 8])
print(ws)

# Singular matrix cause it does not use the second dimension
ws = linear_regression(xs=[(3,1), (5,1), (7,1), (9,1), (11,1)], ys=[4, 4, 6, 8, 8])
print(ws)

# Now with deep learning
ws = linear_regression_dl(xs=[(3,), (5,), (7,), (9,), (11,)], ys=[4, 4, 6, 8, 8])
print(ws)
