from specxplore import importing
import matchms
import pandas as pd

mgf_file_name_pos = "data_import_testing/data/reference_standards_pos/dummy.mgf"
models_and_library_folder_pos = "./data_import_testing/data/ms2query_models_and_library_pos"
models_and_library_folder_neg = "./data_import_testing/data/ms2query_models_and_library_neg"

spectra_list_raw = list(matchms.importing.load_from_mgf(source = mgf_file_name_pos))

spectra_list = importing.clean_spectra(spectra_list_raw)
spectra_list = list(filter(lambda item: item is not None, spectra_list))
print((f"Number of removed spectra: {len(spectra_list_raw) - len(spectra_list)}. "
       f"{len(spectra_list)} spectra remaining."))

missing_structures = 0
for spectrum in spectra_list:
    if spectrum.get("inchi") is None and spectrum.get("smiles") is None and spectrum.get("inchikey") is None:
        missing_structures += 1
print(f"Missing number of structure assignments: {missing_structures}")

spectra_list = spectra_list[0:3]

if False: # prevent time consuming batch run is testing.
    print(importing.get_classes(spectra_list[0].get("inchi"))) # single api call
    classification_df = importing.batch_run_get_classes(spectra_list) # GNPS API Based, Takes a whilte to Run 2 seconds per spectrum at least
    print(classification_df)
    classification_df.to_csv("tmp_test_processing_funs_classifications.csv")
    print(importing.get_classes(spectra_list[0].get("inchi")))
else:
    classification_dd = pd.read_csv("tmp_test_processing_funs_classifications.csv")

print(importing.get_classes(None))

scores_s2v = importing.compute_similarities_s2v(spectra_list, models_and_library_folder_pos)
scores_cos = importing.compute_similarities_cosine(spectra_list, cosine_type="ModifiedCosine")
scores_ms2ds = importing.compute_similarities_ms2ds(spectra_list, models_and_library_folder_pos)

print("Pairwise Similarities Computed")
print(scores_s2v)
print(scores_cos)
print(scores_ms2ds)

print(importing.convert_similarity_to_distance(scores_ms2ds))