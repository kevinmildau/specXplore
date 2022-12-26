# conda activate ms2query 
# conda env ms2query contains installs of ms2query and ms2deepscore with compatible
# matchms verisons.
from specxplore import importing

spectra = importing.load_matchms_spectrum_objects_from_file(
    "data-input/GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")
print("Spectra:", spectra[0:3], len(spectra))
spectra_cleaned = importing.normalize_and_filter_peaks_multiple_spectra(spectra)

print("Cleaned:", spectra_cleaned[0:3], len(spectra_cleaned))

spectra_charge_fixed = importing.add_unknown_charges_to_spectra(
    spectra_cleaned)

print("Charges fixed:", spectra_charge_fixed[0:3], len(spectra_charge_fixed))


