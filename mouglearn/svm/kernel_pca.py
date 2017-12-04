import numpy as np

from kernel import kernel_rbf

class My_KernelPCA:
    def __init__(self, kernel='rbf', hyper=5):
        assert kernel == 'rbf', 'Only rbf for now.'
        self.kernel_param = hyper
        self.kernel = kernel_rbf


    def __compute_gram_matrix(self, x, y):
        n_sample = y.shape[0]
        self.__gram_mat = np.zeros((n_sample, n_sample))
        for i in range(n_sample):
            for j in range(n_sample):
                self.__gram_mat[i, j] = self.kernel(x[i, :], x[j, :], hyper=self.kernel_param)


    def fit(self, x, y, real=True, axis=2):
        self.N = y.shape[0]
        self.__compute_gram_matrix(x, y)
        self.__gram_mat = np.outer(y, y) * self.__gram_mat
        """
        ones = 1/self.N * np.ones((self.N, self.N))
        self.__gram_mat = self.__gram_mat - ones.dot(self.__gram_mat)\
                          - self.__gram_mat.dot(ones) + ones.dot(self.__gram_mat).dot(ones)
        """
        self.__gram_mat -= np.mean(self.__gram_mat)
        self.__gram_mat /= np.var(self.__gram_mat)

        self.eigval, self.eigvec = np.linalg.eig(self.__gram_mat)

        if real:
            self.eigvec = np.real(self.eigvec)
        return self.eigvec[:, :axis]