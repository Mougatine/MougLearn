import math
import random

import numpy as np

def two_borders(noise=1, seed=42):
    np.random.seed(seed)
    size_a = 30
    size_b = 50

    a = np.random.multivariate_normal([0, 5], [[3, 1], [1, 4]], size=[size_a,])
    b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[size_b,])
    X = np.concatenate((a, b))
    y = np.append(np.array([-1 for _ in range(size_a)]),
                np.array([1 for _ in range(size_b)]))


    for i in range(len(y)):
        if random.random() >= noise:
            y[i] *= -1

    return X, y


def multi_aligned(noise=1, seed=42):
    np.random.seed(seed)
    size_a = 50
    size_b = 50
    size_c = 50

    a = np.random.multivariate_normal([20, 0], [[3, 1], [1, 4]], size=[size_a,])
    b = np.random.multivariate_normal([10, 10], [[3, 1], [1, 4]], size=[size_b,])
    c = np.random.multivariate_normal([0, 20], [[3, 1], [1, 3]], size=[size_c,])

    X = np.concatenate((a, b, c))
    y = np.append(np.array([0 for _ in range(size_a)]),
                np.array([1 for _ in range(size_b)]))
    y = np.append(y,
                np.array([2 for _ in range(size_c)]))

    for i in range(len(y)):
        if random.random() > noise:
            y[i] = random.randint(0, 2)

    return X, y


def multiclass_triangle(noise=1, seed=42):
    np.random.seed(seed)
    size_a = 50
    size_b = 50
    size_c = 50

    a = np.random.multivariate_normal([10, 10], [[3, 1], [1, 4]], size=[size_a,])
    b = np.random.multivariate_normal([0, 20], [[3, 1], [1, 4]], size=[size_b,])
    c = np.random.multivariate_normal([0, 0], [[3, 1], [1, 3]], size=[size_c,])

    X = np.concatenate((a, b, c))
    y = np.append(np.array([0 for _ in range(size_a)]),
                np.array([1 for _ in range(size_b)]))
    y = np.append(y,
                np.array([2 for _ in range(size_c)]))

    for i in range(y.shape[0]):
        if random.random() >= noise:
            y[i] = random.randint(0, 2)

    return X, y


def gen_two_moons(d, N, gaussian_noise=False):
    r = 10
    w = 6
    R = (r - w / 2) * np.ones(N) + np.matlib.rand(N) * w
    theta = np.matlib.rand(N) * math.pi
    X = np.concatenate((np.multiply(R, np.cos(theta)), np.multiply(R, np.sin(theta))))
    if gaussian_noise:
        noise = np.random.normal(0,1,N)
        X = np.add(X, noise)
    Y = np.ones(N)
    R = (r - w / 2) * np.ones(N) + np.matlib.rand(N) * w
    theta = -np.matlib.rand(N) * math.pi
    dx = r
    dy = -d
    x = np.concatenate((np.multiply(R, np.cos(theta)) + dx, np.multiply(R, np.sin(theta)) + dy))
    if gaussian_noise:
        noise = np.random.normal(0,1,N)
        x = np.add(x, noise)
    y = -np.ones(N)
    X = np.concatenate((X, x), axis = 1)
    Y = np.concatenate((Y, y))
    seq = np.random.permutation(2 * N)
    X = X[:, seq]
    Y = Y[seq]

    return np.array(X).T, np.array(Y)