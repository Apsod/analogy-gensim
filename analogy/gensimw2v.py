from analogy.wrapper import Wrapper
import numpy
from gensim.models import KeyedVectors


class Gensim(Wrapper):
    """
    An example of the analogy-test wrapper that works for gensims keyed vectors.
    """
    def __init__(self, model):
        self.model = model
        self.w2i = {w: i for i, w in enumerate(model.index2word)}
        self.model.init_sims(replace=True)

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
        return Gensim(KeyedVectors.load_word2vec_format(path))


