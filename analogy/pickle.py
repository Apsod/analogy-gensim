import pickle
from analogy.wrapper import Base


class Wrapper(Base):
    @staticmethod
    def load(path):
        return pickle.load(path)