from analogy.wrapper import Base
import numpy
from gensim.models import KeyedVectors


class Wrapper(Base):
    """
    An example of the analogy-test wrapper that works for gensims keyed vectors.
    """
    def __init__(self, model):
        self.model = model
        self.w2i = {w: i for i, w in enumerate(model.index2word)}
        self.model.init_sims(replace=True)

    def analogies_index(self, queries):
        m = self.model.syn0norm
        aa, bb, xx, yy = zip(*queries)
        N = len(aa)
        ai = [self.w2i[a] for a in aa]
        bi = [self.w2i[b] for b in bb]
        xi = [self.w2i[x] for x in xx]
        yi = [self.w2i[y] for y in yy]
        v = m[bi] + m[xi] - m[ai]
        y = numpy.empty((N, 1))
        sims = v @ m.T
        for i in range(N):
            sims[i, ai[i]] = float('-inf')
            sims[i, bi[i]] = float('-inf')
            sims[i, xi[i]] = float('-inf')
            y[i, 0] = sims[i, yi[i]]

        return [int(c) for c in numpy.less(y, sims).sum(1)]

    def analogies(self, queries):
        m = self.model.syn0norm
        aa, bb, xx = zip(*queries)
        ai = [self.w2i[a] for a in aa]
        bi = [self.w2i[b] for b in bb]
        xi = [self.w2i[x] for x in xx]
        v = m[bi] + m[xi] - m[ai]
        sims = v @ m.T
        for i in range(len(aa)):
            sims[i, ai[i]] = float('-inf')
            sims[i, bi[i]] = float('-inf')
            sims[i, xi[i]] = float('-inf')

        return [self.model.index2word[i] for i in numpy.argmax(sims, axis=1)]

    def members(self, items):
        return [i in self.model for i in items]

    @staticmethod
    def load(path):
        return Wrapper(KeyedVectors.load_word2vec_format(path))


