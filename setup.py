from setuptools import setup

setup(
    name='analogy-gensim',
    version='0.0.1',

    setup_requires=['analogy'],
    install_requires=['analogy'],

    package_dir={'analogy': 'analogy'},
    packages=['analogy'],


)