import numpy
import logging


class Metrics(object):
    def __init__(self, matrix):
        self.matrix = matrix
        self.n2 = (self.matrix * self.matrix).sum(1, keepdims=True)
        self.tt = None

    def get_n2(self, lhs):
        if isinstance(lhs, slice):
            return self.n2[lhs]
        elif isinstance(lhs, numpy.ndarray):
            if self.tt is None:
                return (lhs * lhs).sum(1, keepdims=True)
            else:
                return (lhs @ self.tt * lhs).sum(1, keepdims=True)
        else:
            raise ValueError

    def cosine_similarity(self, lhs, rhs=slice(None)):
        ln = numpy.sqrt(self.get_n2(lhs))
        rn = numpy.sqrt(self.n2[rhs])
        lr = self.dot_product(lhs, rhs)
        lr /= ln
        lr /= rn.T
        return lr

    def euclidean_distance(self, lhs, rhs=slice(None)):
        l2 = self.get_n2(lhs)
        r2 = self.n2[rhs]
        lr = self.dot_product(lhs, rhs)
        return l2 + r2.T - 2*lr

    def dot_product(self, lhs, rhs=slice(None)):
        if isinstance(lhs, slice):
            lm = self.matrix[lhs]
        elif isinstance(lhs, numpy.ndarray):
            lm = lhs
        else:
            raise ValueError

        if self.tt is None:
            return lm @ self.matrix[rhs].T
        else:
            return (lm @ self.tt) @ self.matrix[rhs].T

    def set_transform(self, t):
        self.tt = t.T @ t
        self.n2 = (self.matrix @ self.tt * self.matrix).sum(1, keepdims=True)

    def unset_transform(self):
        self.tt = None
        self.n2 = (self.matrix * self.matrix).sum(1, keepdims=True)


