import pickle
import specxplore.specxplore_data
from specxplore.specxplore_data import specxplore_data, Spectrum
import numpy as np
from specxplore import spectrum_plot

with open("testing/results/phophe_specxplore.pickle", 'rb') as handle:
    data = pickle.load(handle).spectra
spectra = [Spectrum(spec.peaks.mz, max(spec.peaks.mz),idx, spec.peaks.intensities) for idx, spec in enumerate(data)]


figure = spectrum_plot.generate_single_spectrum_plot(spectra[0])
figure.show(renderer = "browser")

figure = spectrum_plot.generate_mirror_plot(spectra[14], spectra[16])
figure.show(renderer = "browser")

#figure = spectrum_plot.generate_multiple_spectra_plot(spectra[14:17])
#figure.show(renderer = "browser")