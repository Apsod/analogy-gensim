from setuptools import setup

setup(
    name='analogy-gensim',
    version='0.0.1',

    requires=['analogy', 'gensim', 'numpy'],
    package_dir={'analogy': 'analogy'},
    packages=['analogy'],


)