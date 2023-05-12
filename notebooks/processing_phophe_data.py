from specxploreImporting import importing
from specxplore.specxplore_data import specxplore_data
from specxploreImporting.importing import attach_metadata, construct_metadata
import matchms
from matchms.importing import load_from_mgf
from matchms.exporting import save_as_mgf
import os
import pandas as pd
import numpy as np
from typing import List
import pickle

# Declare input and output filepaths
models_and_library_pos_path = os.path.join("models", "pos")
raw_experimental_mgf_filepath= os.path.join("data_phophe", "Pos.mgf")
mzmine_annotation_csv_filepath = os.path.join("data_phophe", "Pos_quant_full.csv")

scores_s2v_filename = os.path.join("data_phophe", "output", "s2v.npy")
scores_ms2ds_filename = os.path.join("data_phophe", "output", "ms2ds.npy")
scores_cos_filename = os.path.join("data_phophe", "output", "cos.npy")
output_specxplore_filepath = os.path.join("data_phophe", "output", "phophe.pickle")
# Load raw spectra from standards mgf
spectra = list(load_from_mgf(raw_experimental_mgf_filepath))
print("Length Spectra before processing:", len(spectra))

# Clean spectra
spectra = importing.clean_spectra(spectra)
print("Length Spectra after processing:", len(spectra))

# Read and subselect useful metadata
metadata = pd.read_csv(mzmine_annotation_csv_filepath)
metadata = metadata.loc[:, [
    "id", "rt", "compound_db_identity:compound_name", "compound_db_identity:compound_db_identity",
    "compound_db_identity:mol_formula"]]
metadata.rename(columns={
    "id" : "feature_id",
    "compound_db_identity:compound_name": "chemical_name", 
    "compound_db_identity:compound_db_identity": "standard_id",
    "compound_db_identity:mol_formula" : "molecular_formula"}, inplace=True)
print(metadata)

# Compute pairwise similarity scores
scores_s2v = importing.compute_similarities_s2v(spectra, models_and_library_pos_path)
scores_cos = importing.compute_similarities_cosine(spectra, cosine_type="ModifiedCosine")
scores_ms2ds = importing.compute_similarities_ms2ds(spectra, models_and_library_pos_path)
scores_heuristic = np.maximum(scores_ms2ds, scores_cos)

# Perform kmedoid clustering
k_values = [3, 5, 10]
random_seeds = [int(np.random.randint(0,1000)) for _ in k_values]
similarities = scores_ms2ds
distances = importing.convert_similarity_to_distance(similarities)
kmedoid_list = importing.run_kmedoid_grid(distances, k_values, random_seeds)
#importing.render_kmedoid_fitting_results_in_browser(kmedoid_list)

# Perform tsne embedding
perplexity_values = [x for x in [20]]
random_seeds = [int(np.random.randint(0,1000)) for _ in perplexity_values]
tsne_list_pos = importing.run_tsne_grid(distances, perplexity_values, random_seeds)
#importing.render_tsne_fitting_results_in_browser(tsne_list_pos)

# Construct data for specXplore pickle object

# Metadata Construction
feature_id = [int(spec.get("feature_id")) for spec in spectra]
print(type(feature_id))
specxplore_id = [idx for idx in range(0, len(feature_id))]
is_standard = [True for _ in range(0, len(feature_id))]
specxplore_meta = pd.DataFrame({"specxplore_id":specxplore_id, "feature_id":feature_id})
print(specxplore_meta.dtypes, metadata.dtypes)
specxplore_meta = specxplore_meta.merge(metadata, on= "feature_id", how = "left")

# Class Table construction
kclass_table = pd.DataFrame()
for elem in kmedoid_list:
    kclass_table[elem.k] = elem.cluster_assignments

# Precursor mz value construction
mz = [spec.get("precursor_mz") for spec in spectra]

# Extract Suitable t-SNE coordinate system
tsnedf = pd.DataFrame({"x" : tsne_list_pos[0].x_coordinates, "y" : tsne_list_pos[0].y_coordinates})


tsnedf["is_standard"] = is_standard
tsnedf["id"] = specxplore_id


spectra_converted = importing.convert_matchms_spectra_to_specxplore_spectrum(spectra)
test_data_specxplore = specxplore_data(
  scores_ms2ds,scores_s2v, scores_cos, tsnedf,  kclass_table,
  is_standard, spectra_converted, mz, specxplore_id, specxplore_meta)

print(kclass_table)

with open(output_specxplore_filepath, 'wb') as file:
  pickle.dump(test_data_specxplore, file)