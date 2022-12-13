# run using:
# python3 setup.py build_ext --inplace
#cython: language_level=3
from setuptools import setup
from Cython.Build import cythonize

setup(
    name="specXplore Cython Utils",
    ext_modules=cythonize("utils/cython_utils.pyx")
)