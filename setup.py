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
    name="specXplore setup",
    ext_modules=cythonize(module_paths),
    python_requires='>=3.7',
    install_requires=[
        "dash",
        "cython",
        "plotly",
        "dash_cytoscape",
        "numpy",
        "pandas"
    ]
)
# Cleaning out .cpp files that are not needed after .so object construction.
directories = os.listdir(directory_name)
for item in directories:
    if item.endswith(".cpp"):
         os.remove(os.path.join(directory_name, item))


