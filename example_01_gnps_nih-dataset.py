from specxplore import importing

spectra = importing.load_matchms_spectrum_objects_from_file(
    "data-input/GNPS-NIH-NATURALPRODUCTSLIBRARY.mgf")

print(spectra[1:3])