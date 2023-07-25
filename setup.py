#cython: language_level=3
# python3 setup.py install
from setuptools import setup, Extension
from Cython.Build import cythonize
import os

directory_name = './specxplore'
module_name_list = [
    'egonet_cython', 
    'clustnet_cython', 
    'utils_cython', 
    'datastructures_cython']
module_paths = [
     Extension(
        f'specxplore.{name}',
        sources = [os.path.join(directory_name, name + '.pyx')],
        language = 'c++'
        ) 
     for name in module_name_list]

setup(
    name='specxplore',
    ext_modules=cythonize(module_paths, compiler_directives = {'language_level': '3'}),
    packages=['specxplore'],
    python_requires='>=3.8',
    version = '0.0.0',
    install_requires = [
        'numpy', 
        'ms2query', 
        'matchms>=0.11.0,<=0.13.0',
        'matchmsextras>=0.3.0,<0.4.0',
        'spec2vec>=0.6.0, <=0.7.0',
        'ms2deepscore<0.3.1',
        'h5py',
        'dash',
        'plotly',
        'dash-cytoscape',
        'kmedoids',
        'pandas',
        'cython',
        'scipy',
        'protobuf<=3.20.2', # older version needed for ms2query to be importable
        'dash_daq',
        'dash_bootstrap_components'
        ],
        extras_require={
            'dev': ['pytest']}
)
# Cleaning out .cpp files that are not needed after .so object construction.
directories = os.listdir(directory_name)
for item in directories:
    if item.endswith('.cpp'):
         os.remove(os.path.join(directory_name, item))


