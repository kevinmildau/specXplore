Code for install ms2deepscore into a conda env from within the ms2deepscore repo clone with macos modifications.

```{bash}
conda create —name specxplore38 python=3.8
conda activate specxplore38

pip install lxml 
conda install grpcio
pip install numba
pip install pandas
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
```

Current version: ms2deepscore==0.2.3