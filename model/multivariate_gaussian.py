import numpy as np
import math

class Multivariate_Gaussian(object):
    def __init__(self):
        self._mean = None
        self._cov = None
        self._inv = None
        self._det = None

    def train(self, data):
        self._mean, self._cov, self._inv, self._det = self._train(data):
        return self._mean, self._cov, self._inv, self._det

    def _train(self, data):
        samples = data[:]
        mean = np.mean(samples, axis=0)
        cov = np.cov(samples.T)
        inv = np.linalg.inv(cov) 
        det = np.linalg.det(cov)
        return mean, cov, inv, det

    def pred(self, x):
        mean, cov = self._mean, self._cov
        m = cov.shape[0]
        diff = (x - mean)
        front = np.dot(diff, self._inv)
        fin = np.dot(front, diff.T)
        e = np.exp(-0.5*fin)
        k = 1/ (np.sqrt(self._det)*np.sqrt((2*np.pi)**m))
        return k*e

    @property
    def params(self):
        return self._mean, self._cov
