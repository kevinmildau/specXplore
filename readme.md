# Docstring style for use:
"""
This is a reST style function documentation (restructured text, default for sphinx).

:param param1: this is a first param
:param param2: this is a second param
:returns: this is a description of what is returned
:raises keyError: raises an exception
"""

# ms2query environment creation log
Create ms2query environment with older matchms version that is compatible
with ms2query.

```{bash}
conda create --name ms2query python=3.8
conda activate ms2query

pip install lxml 
conda install grpcio
pip install numba
pip install numba
pip install ipykernel

pip install tensorflow-macos
pip install tensorflow-metal

pip install pkgconfig
pip install pytz
pip install gensim
pip install matchms==0.13.0  
pip install scikit-learn   
pip install spec2vec
pip install matchmsextras

python3 setup.py install (custom ms2query repo)
```

# ms2deepscore environment creation log
Code for install ms2deepscore into a conda env from within the ms2deepscore repo clone with macos modifications.

```{bash}
conda create —name specxplore38 python=3.8
conda activate specxplore38

pip install lxml 
conda install grpcio
pip install numba
pip install numba
pip install ipykernel

pip install tensorflow-macos
pip install tensorflow-metal

python3 setup.py install
```

Additional installs:
```{bash}
pip install spec2vec
pip install dash
pip install dash-cytoscape
pip install plotly-express
pip install kmedoids
pip install sklearn
pip install cython
pip install matchms-extras
pip install matchmsextras
pip install pubchempy 
```

Additional installs for ms2query:
```{bash}
pip install pkgconfig
```

Matchmsextras and matchms-extras seem to be the same thing at different versions...
pip install pubchempy since it is used in matchmsextras, but not installed by default.
Current version: ms2deepscore==0.2.