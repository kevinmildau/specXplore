import pickle
import specxplore.specxplore_data
from specxplore.specxplore_data import specxplore_data
import numpy as np
from specxplore import fragmap

with open("testing/results/phophe_specxplore.pickle", 'rb') as handle:
    data = pickle.load(handle)

spectra = data.spectra
print(len(spectra))
bins = [round(x, 1) for x in list(np.arange(0, 1000 + 0.1, 0.1))]
fig = fragmap.generate_fragmap([0,16,17,18,19], spectra, 0.01, 1, 0, 1000, bins, 0, 200)

fig.show(renderer = "browser")