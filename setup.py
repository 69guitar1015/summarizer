from setuptools import setup, find_packages

setup(
    name='summarizer',
    version='0.0',
    packages=find_packages(),
    install_requires=[
        'chainer',
        'tensorboardX',
    ]
)
