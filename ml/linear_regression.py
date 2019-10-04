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
"""

import numpy as np


