from specxplore import importing
import matchms

spectra_list_raw = list(matchms.importing.load_from_mgf(
    source = "data_import_testing/data/reference_standards_pos/dummy.mgf"))

spectra_list = importing.clean_spectra(spectra_list_raw)
spectra_list = list(filter(lambda item: item is not None, spectra_list))
print((f"Number of removed spectra: {len(spectra_list_raw) - len(spectra_list)}. "
       f"{len(spectra_list)} spectra remaining."))

missing_structures = 0
for spectrum in spectra_list:
    if spectrum.get("inchi") is None and spectrum.get("smiles") is None and spectrum.get("inchikey") is None:
        missing_structures += 1
print(f"Missing number of structure assignments: {missing_structures}")

spectra_list = spectra_list[0:2]

classification_df = importing.batch_run_get_classes(spectra_list) # GNPS API Based, Takes a whilte to Run 2 seconds per spectrum at least
print(classification_df)