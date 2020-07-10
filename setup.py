import os
import sys
from setuptools import setup, find_packages
PACKAGE_NAME = 'URSABench'
MINIMUM_PYTHON_VERSION = 3, 5


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert 0, "'{0}' not found in '{1}'".format(key, module_path)


check_python_version()
setup(
    name='URSABench',
    version=read_package_variable('__version__'),
    description='A PyTorch-based benchmark library for MCMC',
    author='Adam D. Cobb, Meet Vadera, Ben Marlin, Brian Jalaian',
    author_email='cobb.derek.adam@gmail.com, mvadera@cs.umass.edu',
    packages=find_packages(),
    install_requires=['torch>=1.4.0', 'numpy'],
    url='https://github.com/reml-lab/URSABench',
    classifiers=['Development Status :: 1 - Planning', 'License :: OSI Approved :: MIT License', 'Programming Language :: Python :: 3.5'],
    license='MIT',
    keywords='pytorch MCMC BNN',
)
