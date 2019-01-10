import numpy as np
import math

class PCA(object):
    def __init__(self):
        self._w = None
        self._v = None
        self._mean = None
        self._var = None

    def train(self, data):
        self._w, self._v, self._mean, self._var = self._train(data)
        return self._w, self._v, self._mean, self._var

    def _train(self, data):
        samples = data[:]
        mean = np.mean(samples, axis=0)
        samples = samples - mean
        var = np.var(samples, axis=0)
        std_samples = samples/np.sqrt(var)
        cov = np.cov(std_samples.T)
        w, v = LA.eig(np.array(cov))
        return w, v, mean, var

    def pred(self, data):
        return self._project(data, self._v, self._mean, self._var)

    def _project(self, data, v, mean, var):
        samples = data[:]
        samples = samples - mean
        std_samples = samples/np.sqrt(var)
        return std_samples.dot(v)

    @property
    def params(self):
        return self._w, self._v, self._mean, self._var
