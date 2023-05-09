# run using:
# python3 setup.py build_ext --inplace
#cython: language_level=3
from setuptools import setup
from Cython.Build import cythonize
import os

directory_name = "./specxplore"
module_name_list = [
    "egonet_cython.pyx", "clustnet_cython.pyx", "data_transfer_cython.pyx", "specxplore_data_cython.pyx"]
module_paths = [os.path.join(directory_name, name) for name in module_name_list]

setup(
    name="specxplore",
    ext_modules=cythonize(module_paths),
    python_requires='>=3.8',
    version = "0.0.0",
    install_requires=[
        "ms2query",
        "matchms<=0.13",
        "ms2deepscore<0.3.1",
    ]
)
# Cleaning out .cpp files that are not needed after .so object construction.
directories = os.listdir(directory_name)
for item in directories:
    if item.endswith(".cpp"):
         os.remove(os.path.join(directory_name, item))


