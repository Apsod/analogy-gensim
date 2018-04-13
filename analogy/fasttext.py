from analogy.wrapper import Base
from analogy.metrics import Metrics
import numpy
import logging

class Wrapper(Base):
    """
    An example of the analogy-test wrapper that works for fasttext-text formatted vectors.
    """
    def __init__(self, i2w, cm, tm):
        self.i2w = i2w
        self.w2i = {w: i for i, w in enumerate(i2w)}
        self.cm = cm
        self.tm = tm
        self.m = Metrics(cm)
        self.m.set_transform(tm)

    def analogies(self, queries):
        aa, bb, xx = zip(*queries)
        ai = [self.w2i[a] for a in aa]
        bi = [self.w2i[b] for b in bb]
        xi = [self.w2i[x] for x in xx]
        v = self.cm[bi] + self.cm[xi] - self.cm[ai]
        sims = self.m.cosine_similarity(v)
        for i in range(len(aa)):
            sims[i, ai[i]] = float('-inf')
            sims[i, bi[i]] = float('-inf')
            sims[i, xi[i]] = float('-inf')

        return [self.i2w[i] for i in numpy.argmax(sims, axis=1)]

    def members(self, items):
        return [i in self.w2i for i in items]

    @staticmethod
    def load(path):
        cp = '{}.cm'.format(path)
        tp = '{}.tm'.format(path)
        ws = []

        logging.info('Loading context vectors ...')
        with open(cp, 'r') as lines:
            n, d = [int(l) for l in next(lines).split()]
            logging.info('Vocabulary: {}\tDimension: {}'.format(n, d))
            cm = numpy.empty((n, d), dtype='float')
            tm = numpy.empty((n, d), dtype='float')
            for i, line in enumerate(lines):
                w, *xs = line.split()
                ws.append(w)
                cm[i, :] = [float(x) for x in xs]
        logging.info('Loading target vectors ...')
        with open(tp, 'r') as lines:
            assert([n, d] == [int(l) for l in next(lines).split()])
            for i, line in enumerate(lines):
                w, *xs = line.split()
                assert(ws[i] == w)
                tm[i, :] = [float(x) for x in xs]

        logging.info('read')
        return Wrapper(ws, cm, tm)





