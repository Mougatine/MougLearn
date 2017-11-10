import collections

from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from kernel import kernel_linear, kernel_polynomial, kernel_rbf


class My_SVM:
    def __init__(self, C=None, verbose=True, kernel='linear', hyper=None):
        self.C = C
        solvers.options['show_progress'] = verbose
        self.kernel_type = kernel
        self.kernel_param = hyper

        if kernel == 'linear':
            self.kernel = kernel_linear
        elif kernel == 'poly':
            self.kernel = kernel_polynomial
        elif kernel == 'rbf':
            self.kernel = kernel_rbf
        else:
            raise Exception('The kernel {} is not known.'.format(kernel))
        
    
    def __compute_gram_matrix(self, x, y):
        self.__gram_mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.__gram_mat[i, j] = self.kernel(x[i, :], x[j, :], hyper=self.kernel_param)



    def __solve_dual(self, y):
        P = matrix(np.outer(y, y) * self.__gram_mat)
        if self.C is None:
            G = matrix(-np.eye(self.N))
            h = matrix(np.zeros(self.N))
        else:
            G = matrix(np.vstack((-np.eye(self.N), np.identity(self.N))))
            h = matrix(np.hstack((np.zeros(self.N), np.ones(self.N) * self.C)))
        q = matrix(-np.ones(self.N))
        A = matrix(y.reshape(1, -1).astype(float))
        b = matrix(0.0)
        result = solvers.qp(P, q, G, h, A, b)
        if result['status'] != 'optimal':
            print('No optimal solution found.')
            
        self.__lagrange_coefs = np.ravel(result['x'])
        self.__lagrange_coefs[np.where(self.__lagrange_coefs < 1e-5)[0]] = 0


    def train(self, x, y):
        self.x = x
        self.N = x.shape[0]
        self.D = x.shape[1]
        self.__compute_gram_matrix(x, y)
        self.__solve_dual(y)

        if self.C is None:
            sv = np.where(self.__lagrange_coefs > 0)[0]
            not_sv = np.where(self.__lagrange_coefs <= 0)[0]
        else:
            sv = np.where((self.__lagrange_coefs > 0) & (self.__lagrange_coefs <= self.C))[0]
            not_sv = np.where((self.__lagrange_coefs <= 0) | (self.__lagrange_coefs > self.C))[0]
        self.__alphas = self.__lagrange_coefs[sv]
        self.sv_x = x[sv]
        self.sv_y = y[sv]
        self.not_sv_x = x[not_sv]
        self.not_sv_y = y[not_sv]
        
        self.__W = None
        if self.kernel_type == 'linear':
            self.__W = np.zeros(self.D)
            for i in range(self.__alphas.shape[0]):
                self.__W += self.__alphas[i] * np.dot(self.sv_y[i], self.sv_x[i])
            
        self.__bias = 0
        for i in range(self.__alphas.shape[0]):
            self.__bias += self.sv_y[i]
            self.__bias -= np.sum(self.__alphas * self.sv_y * self.__gram_mat[sv[i], sv])
        self.__bias /= self.__alphas.shape[0] if self.__alphas.shape[0] != 0 else 1


    def decision(self, x):
        if self.kernel_type == 'linear':
            return np.dot(x, self.__W) + self.__bias
        else:
            y_predict = np.zeros(x.shape[0])
            for i in range(x.shape[0]):
                s = 0
                for a, sv_y, sv_x in zip(self.__alphas, self.sv_y, self.sv_x):
                    s += a * sv_y * self.kernel(x[i, :], sv_x, hyper=self.kernel_param)
                y_predict[i] = s
            return y_predict + self.__bias

    
    def process(self, x):
        return np.sign(self.decision(x))
    
    
    def bias_get(self):
        return self.__bias
    
    
    def weights_get(self):
        return self.__W
    
    
    def lagrange_coeffs(self):
        return self.__lagrange_coefs
    
    
    def score(self, x, y):
        return np.sum(self.process(x) == y) / y.shape[0]


    def print_2Ddecision(self, print_sv=True, print_non_sv=False):
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

        if print_sv:
            colors = ['red' if self.sv_y[i] == -1 else 'blue'
                      for i in range(self.sv_x.shape[0])]
            plt.scatter(self.sv_x[:, 0], self.sv_x[:, 1], color=colors)
        if print_non_sv:
            colors = ['red' if self.not_sv_y[i] == -1 else 'blue'
                      for i in range(self.not_sv_x.shape[0])]
            plt.scatter(self.not_sv_x[:, 0], self.not_sv_x[:, 1], color=colors,
                        facecolor='none')


        levels = [-1, 0, 1]
        ZS = self.decision(np.c_[XX.ravel(), YY.ravel()])
        ZS = ZS.reshape(XX.shape)
        CS = plt.contour(XX, YY, ZS, levels,
                 origin='lower',
                 linewidths=2,
                 extent=(-3, 3, -2, 2))
        for line, level in zip(CS.collections, levels):
            if level == 0:
                line.set_linestyle([(None, None)])
            else:
                line.set_linestyle([(0, (2.0, 3.0))])
        plt.clabel(CS, inline=False, fontsize=0)
        """
        x = np.linspace(xmin, xmax, 100)
        y = np.linspace(ymin, ymax, 100)

        (x_b, y_b) = np.meshgrid(x,y)
        z_b = np.zeros(x_b.shape)
        for i in range(0, x_b.shape[0]):
            for j in range(0, x_b.shape[1]):
                z_b[i,j] = self.decision((np.array([[x_b[i,j]],[y_b[i,j]]])))[0]

        cdict= [(0.,"red"), (0.5,"white"), (1., "black")]
        RedBlack = LinearSegmentedColormap.from_list("RedBlack", cdict)
        plt.pcolor(x_b,y_b,z_b, cmap=RedBlack)
        """
        plt.show()