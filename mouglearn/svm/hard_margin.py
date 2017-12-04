from cvxopt import matrix, solvers
import matplotlib.pyplot as plt
import numpy as np

class MyHardMarginSVM:
        
    def __compute_gram_matrix(self, x, y):
        self.__gram_mat = np.zeros((self.N, self.N))
        for i in range(self.N):
            for j in range(self.N):
                self.__gram_mat[i, j] = y[i] * y[j] * np.dot(x[i, :], x[j, :])
        
        
    def train(self, x, y):
        self.x = x
        self.N = x.shape[0]
        self.D = x.shape[1]
        self.__compute_gram_matrix(x, y)
        
        P = matrix(self.__gram_mat)
        G = matrix(-np.eye(self.N))
        q = matrix(-np.ones(self.N))
        h = matrix(np.zeros(self.N))
        A = matrix(y.reshape(1, -1).astype(float))
        b = matrix(0.0)
        result = solvers.qp(P, q, G, h, A, b)
        if result['status'] != 'optimal':
            print("Can't solve the problem.")
            return
            
        self.__alphas = np.ravel(result['x'])
        self.__lagrange_coef = self.__alphas
        self.__lagrange_coef[np.where(self.__lagrange_coef < 1e-5)[0]] = 0
        sv = self.__alphas > 1e-5
        not_sv = self.__alphas < 1e-5
        self.__alphas = self.__alphas[sv]
        
        self.sv_x = x[sv]
        self.sv_y = y[sv]
        self.not_sv_x = x[not_sv]
        self.not_sv_y = y[not_sv]
        
        self.__W = np.zeros(self.D)
        for i in range(len(self.__alphas)):
            self.__W += self.__alphas[i] * self.sv_y[i] * self.sv_x[i]

        self.__bias = 0
        self.__bias = np.mean(self.sv_y - self.decision(self.sv_x))
        
        
    def decision(self, x):
        return np.dot(x, self.__W) + self.__bias
    
    
    def process(self, x):
        return np.sign(self.decision(x))
    
    
    def bias_get(self):
        return self.__bias
    
    
    def weights_get(self):
        return self.__W
    
    
    def lagrange_coeffs(self):
        return self.__lagrange_coef
    
    
    def print_2Ddecision(self, bounds, print_sv=True, print_non_sv=False):
        xmin, ymin, xmax, ymax = bounds
        x = range(xmin, xmax+1)

        slope = -self.__W[0] / self.__W[1]
        intercep = -self.__bias / self.__W[1]
        plt.plot(x, slope* x + intercep)
        
        plt.ylim(ymin=ymin, ymax=ymax)
        plt.xlim(xmin=xmin, xmax=xmax)
        
        XX, YY = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        Z = self.process(np.c_[XX.ravel(), YY.ravel()])
        Z = Z.reshape(XX.shape)
        plt.pcolormesh(XX, YY, Z, cmap=plt.cm.Paired)

        if print_sv:
            colors = ['red' if self.sv_y[i] == -1 else 'blue'
                      for i in range(self.sv_x.shape[0])]
            plt.scatter(self.sv_x[:, 0], self.sv_x[:, 1], color=colors)
        if print_non_sv:
            colors = ['red' if self.not_sv_y[i] == -1 else 'blue'
                      for i in range(self.not_sv_x.shape[0])]
            plt.scatter(self.not_sv_x[:, 0], self.not_sv_x[:, 1], color=colors,
                        facecolor='none')

        margin = 1 / np.linalg.norm(self.__W)
        plt.plot(x, slope * x + intercep - margin, color='black', linestyle=':')
        plt.plot(x, slope * x + intercep + margin, color='black', linestyle=':')

        plt.show()