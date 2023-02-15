import pickle
import specxplore.specxplore_data
from specxplore.specxplore_data import specxplore_data, Spectrum
import numpy as np
from specxplore import fragmap
from specxplore.compose import compose_function
from functools import partial
import plotly.graph_objects as go
with open("data_import_testing/results/phophe_specxplore.pickle", 'rb') as handle:
    data = pickle.load(handle).spectra

spectra = [Spectrum(spec.peaks.mz, max(spec.peaks.mz),idx, spec.peaks.intensities) for idx, spec in enumerate(data)]
#print(f"Spectrum object 0: {spectra[0]}")

step = 0.1
spectrum_bin_template = [round(x, 1) for x in list(np.arange(0, 1000 + step, step))]

# test binning
bin_test = fragmap.bin_spectrum(spectra[0], spectrum_bin_template)
#print(f"Binning Output: {bin_test}")

#
spectra_subset = [spectra[idx] for idx in [0,16,17,18]]
binned_spectra = [fragmap.bin_spectrum(spectrum, spectrum_bin_template) for spectrum in spectra_subset]
#print("--> Multiple binning output:", binned_spectra)

#
spectrum_df = fragmap.spectrum_list_to_pandas(binned_spectra)
#print("--> Spectrum df generator output\n", spectrum_df)

# test prevalence filter
#print(all(spectrum_df == fragmap.generate_prevalence_filtered_df(spectrum_df, 1)))
#print(fragmap.generate_prevalence_filtered_df(spectrum_df, 2))

# test mz filter
#print(fragmap.generate_mz_range_filtered_df(spectrum_df, 200,1000))

# test intensity filter
#print(fragmap.generate_intensity_filtered_df(spectrum_df, 0.9))


#print("Checkpoint 1")
tmp = fragmap.generate_intensity_filtered_df(spectrum_df, intensity_min=0.01)
#print("Checkpoint 2", tmp)
tmp = fragmap.generate_mz_range_filtered_df(tmp, mz_min = 0, mz_max = 1000)
#print("Checkpoint 3", tmp)
tmp = fragmap.generate_prevalence_filtered_df(tmp, n_min_occurrences = 2)
#print("Checkpoint 4", tmp)


# test composed filter pipeline
#filter_pipeline = compose_function(
#    partial(fragmap.generate_intensity_filtered_df, intensity_min=0.01),
#    partial(fragmap.generate_mz_range_filtered_df, 
#        mz_min = 0, mz_max = 1000),
#    partial(fragmap.generate_prevalence_filtered_df, n_min_occurrences = 2))

##print(filter_pipeline(None))
#tmp_df = filter_pipeline(spectrum_df)
##print(tmp_df)




fig = fragmap.generate_fragmap(spectra, 0.0,2,0,1000,spectrum_bin_template,0,200)

fig.show(renderer = "browser")


# 3 conditions that should be tested in generate fragmap panel
# 1 --> empty or too small input; return empty
# 2 --> input leads to fragmap build, return fragmap
# 3 --> input is valid, but filtering prevents fragmap built, return empty
#   --> capture the various error points where this could be a problem