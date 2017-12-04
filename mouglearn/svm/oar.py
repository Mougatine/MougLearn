from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np

class OAR_SVM:
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
        self.nb_classes = self.classes.shape[0]
        self.__alphas = [None for _ in range(self.nb_classes)]
        self.__lagrange_coef = [None for _ in range(self.nb_classes)]
        self.sv_x = [None for _ in range(self.nb_classes)]
        self.sv_y = [None for _ in range(self.nb_classes)]
        self.not_sv_x = [None for _ in range(self.nb_classes)]
        self.not_sv_y = [None for _ in range(self.nb_classes)]
        self.__W = [None for _ in range(self.nb_classes)]
        self.__bias = [None for _ in range(self.nb_classes)]
        self.__gram_mat = [None for _ in range(self.nb_classes)]

        for idx in range(self.nb_classes):
            assign_class = lambda v: -1 if v == self.classes[idx] else 1
            y_tmp = np.array([assign_class(y[i]) for i in range(y.shape[0])])
            self.__train(x, y_tmp, idx)
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
        return [self.__decision(x, idx) for idx in range(self.nb_classes)]


    def __decision(self, x, idx):
        return np.dot(x, self.__W[idx]) + self.__bias[idx]


    def process(self, x):
        decisions = [np.sign(d) for d in self.decision(x)]
        preds = np.zeros(x.shape[0])
        for elt_idx in range(x.shape[0]):
            for cls_idx in range(self.nb_classes):
                if decisions[cls_idx][elt_idx] == -1:
                    preds[elt_idx] = self.classes[cls_idx]
                    break
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

        XX, YY = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        Z = self.process(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Pastel2)

        for idx in range(self.nb_classes):
            slope = -self.__W[idx][0] / self.__W[idx][1]
            intercep = -self.__bias[idx] / self.__W[idx][1]
            plt.plot(x, slope * x + intercep, color=color_map[idx])

        colors = [color_map[self.y[i]] for i in range(self.x.shape[0])]
        plt.scatter(self.x[:, 0], self.x[:, 1], color=colors)



        plt.show()