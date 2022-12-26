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