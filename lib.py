import numpy as np

# X -> (m, n)
# t -> (n, 1)
# y -> (m, 1)

cost = lambda X, y, t: ((X @ t - y) ** 2).sum() / len(y)
grad = lambda X, y, t: 2 * X.T @ (X @ t - y) / len(y)
