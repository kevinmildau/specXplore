import pickle
import specxplore.specxplore_data
from specxplore.specxplore_data import specxplore_data, Spectrum
import numpy as np
from specxplore import fragmap

with open("data_import_testing/results/phophe_specxplore.pickle", 'rb') as handle:
    data = pickle.load(handle).spectra

spectra = [Spectrum(spec.peaks.mz, max(spec.peaks.mz),idx, spec.peaks.intensities) for idx, spec in enumerate(data)]
print(f"Spectrum object 0: {spectra[0]}")

spectrum_bin_template = [round(x, 1) for x in list(np.arange(0, 1000 + 10, 10))]

# test binning
bin_test = fragmap.bin_spectrum(spectra[0], spectrum_bin_template)
print(f"Binning Output: {bin_test}")

#
spectra_subset = [spectra[idx] for idx in [0,16,17,18]]
binned_spectra = [fragmap.bin_spectrum(spectrum, spectrum_bin_template) for spectrum in spectra_subset]
print("--> Multiple binning output:", binned_spectra)

#
spectrum_df = fragmap.spectrum_list_to_pandas(binned_spectra)
print("--> Spectrum df generator output\n", spectrum_df)
