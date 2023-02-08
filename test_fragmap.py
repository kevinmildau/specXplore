import pickle
import specxplore.specxplore_data
from specxplore.specxplore_data import specxplore_data, Spectrum
import numpy as np
from specxplore import fragmap

with open("data_import_testing/results/phophe_specxplore.pickle", 'rb') as handle:
    data = pickle.load(handle).spectra

spectra = [Spectrum(spec.peaks.mz, max(spec.peaks.mz),idx, spec.peaks.intensities) for idx, spec in enumerate(data)]

print(spectra[0])
print(len(spectra))

spectrum_df = fragmap.spectrum_list_to_pandas(spectra[0:2])
print("--> Spectrum df", spectrum_df)
spectrum_bin_template = [round(x, 1) for x in list(np.arange(0, 1000 + 1, 1))]
print("--> Spectrum bin template head:", spectrum_bin_template[0:6])
spectrum_df_binned = fragmap.bin_spectra(spectrum_df, spectrum_bin_template)
print("--> Spectrum df after binning:", spectrum_df_binned)
#bins = [round(x, 1) for x in list(np.arange(0, 1000 + 0.1, 0.1))]
#fig = fragmap.generate_fragmap([0,16,17,18,19], spectra, 0.01, 1, 0, 1000, bins, 0, 200)

#fig.show(renderer = "browser")