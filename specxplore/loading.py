
import pickle
import pandas as pd

def load_pairwise_sim_matrices():
    """Crude tmp function to load matrices until on disk functionality and
    naming convections established."""
    file = open("data/sim_matrix_ms2deepscore", 'rb')
    sm_ms2deepscore = pickle.load(file)
    file.close()
    file = open("data/sim_matrix_cosine_modified_extracted.pickle", 'rb')
    sm_cosine_modified = pickle.load(file)
    file.close()
    file = open("data/sim_matrix_spec2vec", 'rb')
    sm_spec2vec= pickle.load(file)
    file.close()
    return sm_ms2deepscore, sm_cosine_modified, sm_spec2vec

def process_structure_class_table(file_location):
    """Helper function to parse class table."""
    tmp = pd.read_csv("data/classification_table.csv", index_col=False)
    structure_dict = dict(inchi_key = list(tmp["inchi_key"]), smiles = list(tmp["smiles"]))
    #print(structure_dict.keys())
    tmp = tmp.drop(["inchi_key", "smiles", "idx"], axis = 1)
    class_dict = {elem : list(tmp[elem]) for elem in tmp.columns}
    #print(class_dict.keys())
    #print(class_dict["cf_class"])
    return structure_dict, class_dict

def extract_classes_from_ms2query_results(file_location):
    """Helper function to parse class table."""
    tmp = pd.read_csv(file_location, index_col=False)
    #print(tmp, tmp.columns)
    tmp.fillna("Unknown", inplace=True)
    #print(structure_dict.keys())
    tmp = tmp.drop([
        'query_spectrum_nr', 'ms2query_model_prediction',
        'precursor_mz_difference', 'precursor_mz_query_spectrum',
        'precursor_mz_analog', 'inchikey', 'spectrum_ids',
        'analog_compound_name', 'retention_time', 'retention_index', 'smiles'], 
        axis = 1)
    tmp.replace(' ', '_', regex=True)
    class_dict = {elem : list(tmp[elem]) for elem in tmp.columns}
    #print(class_dict.keys())
    #print(class_dict["cf_class"])
    return class_dict