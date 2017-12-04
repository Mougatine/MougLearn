import numpy as np


def kernel_linear(x, y, *, hyper=None):
    return np.dot(x, y)


def kernel_polynomial(x, y, *, hyper):
    return (np.dot(x, y) + 1) ** hyper


def kernel_rbf(x, y, *, hyper):
    return np.exp(-0.5 * (np.linalg.norm(x - y) ** 2) / hyper ** 2)