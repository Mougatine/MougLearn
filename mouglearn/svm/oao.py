import itertools as its

from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np

class OAO_SVM:
    def __init__(self, *, C=1.0):
        self.C = C
        solvers.options['show_progress'] = False


    def __compute_gram_matrix(self, x, y, idx):
        self.__gram_mat[idx] = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.__gram_mat[idx][i, j] = y[i] * y[j] * np.dot(x[i, :], x[j, :])


    def train(self, x, y):
        self.x = x
        self.y = y

        print('Training...', end=' ')
        self.classes = np.unique(y)
        self.pair_classes = list(its.combinations(self.classes, r=2))
        self.nb_pair = len(self.pair_classes)

        self.__alphas = [None for _ in range(self.nb_pair)]
        self.__lagrange_coef = [None for _ in range(self.nb_pair)]
        self.sv_x = [None for _ in range(self.nb_pair)]
        self.sv_y = [None for _ in range(self.nb_pair)]
        self.not_sv_x = [None for _ in range(self.nb_pair)]
        self.not_sv_y = [None for _ in range(self.nb_pair)]
        self.__W = [None for _ in range(self.nb_pair)]
        self.__bias = [None for _ in range(self.nb_pair)]
        self.__gram_mat = [None for _ in range(self.nb_pair)]

        for i, (cls1, cls2) in enumerate(self.pair_classes):
            idx = np.where((y == cls1) | (y == cls2))[0]
            y_tmp = np.array([-1 if y[idx][i] == cls1 else 1
                              for i in range(y[idx].shape[0])])

            self.__train(x[idx], y_tmp, i)
        print('Done!')


    def __train(self, x, y, idx):
        self.N = x.shape[0]
        self.D = x.shape[1]
        self.__compute_gram_matrix(x, y, idx)

        P = matrix(self.__gram_mat[idx])
        G = matrix(np.vstack((-np.eye(self.N), np.identity(self.N))))
        q = matrix(-np.ones(self.N))
        h = matrix(np.hstack((np.zeros(self.N), np.ones(self.N) * self.C)))
        A = matrix(y.reshape(1, -1).astype(float))
        b = matrix(0.0)
        result = solvers.qp(P, q, G, h, A, b)
        if result['status'] != 'optimal':
            print("Can't solve the problem.")
            return

        self.__alphas[idx] = np.ravel(result['x'])
        self.__lagrange_coef[idx] = self.__alphas[idx]
        self.__lagrange_coef[idx][np.where(self.__lagrange_coef[idx] < 1e-5)[0]] = 0
        sv = np.where((self.__alphas[idx] > 1e-5) & (self.__alphas[idx] <= self.C))[0]
        not_sv = np.where((self.__alphas[idx] < 1e-5) | (self.__alphas[idx] > self.C))[0]
        self.__alphas[idx] = self.__alphas[idx][sv]

        self.sv_x[idx] = x[sv]
        self.sv_y[idx] = y[sv]
        self.not_sv_x[idx] = x[not_sv]
        self.not_sv_y[idx] = y[not_sv]

        self.__W[idx] = np.zeros(self.D)
        for i in range(len(self.__alphas[idx])):
            self.__W[idx] += self.__alphas[idx][i] * self.sv_y[idx][i] * self.sv_x[idx][i]

        self.__bias[idx] = 0
        self.__bias[idx] = np.mean(self.sv_y[idx] - self.__decision(self.sv_x[idx], idx))


    def decision(self, x):
        return [self.__decision(x, idx) for idx in range(self.nb_pair)]


    def __decision(self, x, idx):
        return np.dot(x, self.__W[idx]) + self.__bias[idx]


    def process(self, x):
        decisions = [np.sign(d) for d in self.decision(x)]
        preds = np.zeros(x.shape[0])
        for elt_idx in range(x.shape[0]):
            vote_counter = np.zeros(self.classes.shape[0])
            for i, (cls1, cls2) in enumerate(self.pair_classes):
                if decisions[i][elt_idx] == -1:
                    vote_counter[np.where(self.classes == cls1)[0]] += 1
                else:
                    vote_counter[np.where(self.classes == cls2)[0]] += 1
            preds[elt_idx] = self.classes[vote_counter.argmax()]

        return preds


    def bias_get(self):
        return self.__bias


    def weights_get(self):
        return self.__W


    def lagrange_coeffs(self):
        return self.__lagrange_coef


    def print_2Ddecision(self, bounds):
        xmin, ymin, xmax, ymax = bounds
        plt.ylim(ymin=ymin, ymax=ymax)
        plt.xlim(xmin=xmin, xmax=xmax)
        x = range(xmin, xmax+1)
        color_map = ['red', 'blue', 'green']
        color_sep = ['black', 'purple', 'teal']

        XX, YY = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        Z = self.process(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Pastel2)

        for idx in range(self.nb_pair):
            slope = -self.__W[idx][0] / self.__W[idx][1]
            intercep = -self.__bias[idx] / self.__W[idx][1]
            plt.plot(x, slope * x + intercep, color=color_sep[idx])

        colors = [color_map[self.y[i]] for i in range(self.x.shape[0])]
        plt.scatter(self.x[:, 0], self.x[:, 1], color=colors)

        plt.show()