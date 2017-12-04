import collections
import itertools

from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from kernel import kernel_linear, kernel_polynomial, kernel_rbf


class My_SVM:
    def __init__(self, C=None, verbose=True, kernel='linear', hyper=None,
                 multiclass='oao'):
        self.C = C
        solvers.options['show_progress'] = verbose
        self.kernel_type = kernel
        self.kernel_param = hyper

        kernels = {'linear': kernel_linear, 'poly': kernel_polynomial,
                   'rbf': kernel_rbf}
        try:
            self.kernel = kernels[kernel]
        except:
            raise Exception('The kernel `{}` is not known.'.format(kernel))

        if multiclass != 'oao' and multiclass != 'oar':
            raise Exception('The multiclass scheme `{}` is not known.'.format(multiclass))
        self.multiclass = multiclass


    def __compute_gram_matrix(self, x, y, idx):
        n_sample = y.shape[0]
        self.__gram_mat[idx] = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            for j in range(n_sample):
                self.__gram_mat[idx][i, j] = self.kernel(x[i, :], x[j, :], hyper=self.kernel_param)


    def __generate_array(self, size, default=None):
        return [default for _ in range(size)]


    def train(self, x, y):
        self.x = x
        self.y = y
        self.N = x.shape[0]
        self.D = x.shape[1]
        self.classes = np.unique(y)

        if self.multiclass == 'oao':
            self.pair_classes = list(itertools.combinations(self.classes, r=2))
            self.nb_pair = len(self.pair_classes)
            self.size = self.nb_pair
        elif self.multiclass == 'oar':
            self.size = len(self.classes)

        self.__alphas = self.__generate_array(self.size)
        self.__lagrange_coefs = self.__generate_array(self.size)
        self.sv_x = self.__generate_array(self.size)
        self.sv_y = self.__generate_array(self.size)
        self.not_sv_x = self.__generate_array(self.size)
        self.not_sv_y = self.__generate_array(self.size)
        self.__W = self.__generate_array(self.size)
        self.__bias = self.__generate_array(self.size)
        self.__gram_mat = self.__generate_array(self.size)

        if self.multiclass == 'oao':
            for i, (cls1, cls2) in enumerate(self.pair_classes):
                idx = np.where((y == cls1) | (y == cls2))[0]

                y_tmp = np.empty_like(y[idx])
                y_tmp[:] = y[idx]
                y_tmp[np.where(y_tmp == cls1)] = -1
                y_tmp[np.where(y_tmp == cls2)] = 1

                self.__train(x[idx], y_tmp, i)
        elif self.multiclass == 'oar':
            for i in range(len(self.classes)):
                assign_class = lambda v: -1 if v == self.classes[i] else 1
                y_tmp = np.array([assign_class(y[j]) for j in range(y.shape[0])])

                self.__train(x, y_tmp, i)


    def __solve_dual(self, y, idx):
        n_sample = y.shape[0]
        P = matrix(np.outer(y, y) * self.__gram_mat[idx])
        if self.C is None:
            G = matrix(-np.eye(n_sample))
            h = matrix(np.zeros(n_sample))
        else:
            G = matrix(np.vstack((-np.eye(n_sample), np.identity(n_sample))))
            h = matrix(np.hstack((np.zeros(n_sample), np.ones(n_sample) * self.C)))
        q = matrix(-np.ones(n_sample))
        A = matrix(y.reshape(1, -1).astype(float))
        b = matrix(0.0)
        result = solvers.qp(P, q, G, h, A, b)
        if result['status'] != 'optimal':
            print('No optimal solution found.')

        self.__lagrange_coefs[idx] = np.ravel(result['x'])
        self.__lagrange_coefs[idx][np.where(self.__lagrange_coefs[idx] < 1e-5)[0]] = 0


    def __train(self, x, y, idx):
        self.__compute_gram_matrix(x, y, idx)
        self.__solve_dual(y, idx)

        if self.C is None:
            sv = np.where(self.__lagrange_coefs[idx] > 0)[0]
            not_sv = np.where(self.__lagrange_coefs[idx] <= 0)[0]
        else:
            sv = np.where((self.__lagrange_coefs[idx] > 0) & (self.__lagrange_coefs[idx] <= self.C))[0]
            not_sv = np.where((self.__lagrange_coefs[idx] <= 0) | (self.__lagrange_coefs[idx] > self.C))[0]
        self.__alphas[idx] = self.__lagrange_coefs[idx][sv]
        self.sv_x[idx] = x[sv]
        self.sv_y[idx] = y[sv]
        self.not_sv_x[idx] = x[not_sv]
        self.not_sv_y[idx] = y[not_sv]

        if self.kernel_type == 'linear':
            self.__W[idx] = np.zeros(self.D)
            for i in range(self.__alphas[idx].shape[0]):
                self.__W[idx] += self.__alphas[idx][i] * np.dot(self.sv_y[idx][i], self.sv_x[idx][i])

        self.__bias[idx] = 0
        for i in range(self.__alphas[idx].shape[0]):
            self.__bias[idx] += self.sv_y[idx][i]
            self.__bias[idx] -= np.sum(self.__alphas[idx] * self.sv_y[idx] * self.__gram_mat[idx][sv[i], sv])
        self.__bias[idx] /= self.__alphas[idx].shape[0] if self.__alphas[idx].shape[0] != 0 else 1


    def __decision(self, x, idx):
        if self.kernel_type == 'linear':
            return np.dot(x, self.__W[idx]) + self.__bias[idx]
        else:
            y_predict = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                s = 0
                for a, sv_y, sv_x in zip(self.__alphas[idx], self.sv_y[idx], self.sv_x[idx]):
                    s += a * sv_y * self.kernel(x[i, :], sv_x, hyper=self.kernel_param)
                y_predict[i] = s
            return y_predict + self.__bias[idx]


    def decision(self, x):
        return [self.__decision(x, idx) for idx in range(self.size)]


    def process(self, x):
        decisions = [np.sign(d) for d in self.decision(x)]
        placeholder = len(self.classes)+1
        preds = np.full(x.shape[0], placeholder)

        if self.multiclass == 'oao':
            for elt_idx in range(x.shape[0]):
                vote_counter = np.zeros(len(self.classes))
                for i, (cls1, cls2) in enumerate(self.pair_classes):
                    if decisions[i][elt_idx] == -1:
                        vote_counter[np.where(self.classes == cls1)[0]] += 1
                    else:
                        vote_counter[np.where(self.classes == cls2)[0]] += 1

                if len(vote_counter[np.where(vote_counter == 1)[0]]) == len(vote_counter):
                    preds[elt_idx] = len(self.classes)
                else:
                    preds[elt_idx] = self.classes[vote_counter.argmax()]
        elif self.multiclass == 'oar':
            for elt_idx in range(x.shape[0]):
                for cls_idx in range(len(self.classes)):
                    if decisions[cls_idx][elt_idx] == -1:
                        if preds[elt_idx] == placeholder: # Never seen
                            preds[elt_idx] = self.classes[cls_idx]
                        else:
                            preds[elt_idx] = len(self.classes) # Not predictable

        preds[np.where(preds == placeholder)[0]] = len(self.classes)
        return preds



    def bias_get(self):
        return self.__bias


    def weights_get(self):
        return self.__W


    def lagrange_coeffs(self):
        return self.__lagrange_coefs


    def score(self, x, y):
        return np.sum(self.process(x) == y) / y.shape[0]


    def print_2Ddecision(self, levels=[-1, 0, 1]):
        xmin = np.amin(self.x[:, 0]) - 1
        xmax = np.amax(self.x[:, 0]) + 1
        ymin = np.amin(self.x[:, 1]) - 1
        ymax = np.amax(self.x[:, 1]) + 1

        plt.xlim(xmin=xmin, xmax=xmax)
        plt.ylim(ymin=ymin, ymax=ymax)

        XX, YY = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        Z = self.process(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z, cmap='Pastel1')

        if levels:
            ZS = self.decision(np.c_[XX.ravel(), YY.ravel()])
            for idx in range(self.size):
                ZS_i = ZS[idx].reshape(XX.shape)
                CS = plt.contour(XX, YY, ZS_i, levels,
                        origin='lower',
                        linewidths=2,
                        extent=(-3, 3, -2, 2))
                for line, level in zip(CS.collections, levels):
                    if level == 0:
                        line.set_linestyle([(None, None)])
                    else:
                        line.set_linestyle([(0, (2.0, 3.0))])
                plt.clabel(CS, inline=False, fontsize=0)

        if len(self.classes) == 2:
            # Support Vectors
            colors = ['red' if self.sv_y[0][i] == -1 else 'blue'
                    for i in range(self.sv_x[0].shape[0])]
            plt.scatter(self.sv_x[0][:, 0], self.sv_x[0][:, 1], color=colors)

            # Non-Support Vectors
            colors = ['red' if self.not_sv_y[0][i] == -1 else 'blue'
                    for i in range(self.not_sv_x[0].shape[0])]
            plt.scatter(self.not_sv_x[0][:, 0], self.not_sv_x[0][:, 1], color=colors,
                        facecolor='none')
        else:
            colors_map = ['red', 'green', 'blue']
            colors = [colors_map[self.y[i]] for i in range(len(self.y))]
            plt.scatter(self.x[:, 0], self.x[:, 1], color=colors)

        plt.show()