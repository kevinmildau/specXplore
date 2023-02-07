from typing import List, Union, Dict
import matchms.utils # UPDATE ALERT: this may not be the same in matchms >0.14
import json
import urllib
import time
from matchms.typing import SpectrumType
import os
import gensim
from spec2vec import Spec2Vec
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine, CosineHungarian
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
import copy
import numpy as np
from functools import reduce, partial
from matchms.filtering import (default_filters, repair_inchi_inchikey_smiles, derive_inchikey_from_inchi, 
    derive_smiles_from_inchi, derive_inchi_from_smiles, harmonize_undefined_inchi, harmonize_undefined_inchikey,
    harmonize_undefined_smiles, normalize_intensities, select_by_mz, reduce_to_number_of_peaks)

def get_classes(inchi: str):
    """Function returns cf (classyfire) and npc (natural product classifier) 
    classes for a provided inchi.
    
    Sould no information be available, the output will be a list of 
    "Not Available" strings of length 9 for 5 cf classes and 4 npc classes for
    downstream conversion to pandas data frame compatibility.

    Context: Effectively fetches and constructs single row of pandas df using
    a list of dictionaries.

    :param inchi: A valid inchi for which class information should be fetched. An input of "" is handles as an 
    exception with a dict of "Not Available" data being returned.
    """
    key_list = ['inchi_key', 'smiles', 'cf_kingdom', 'cf_superclass',
        'cf_class', 'cf_subclass', 'cf_direct_parent', 'npc_class', 
        'npc_superclass', 'npc_pathway', 'npc_isglycoside']
    if inchi == "":
        print("No inchi, returning Not Available structure.")
        output = {key:"Not Available" for key in key_list}
        return output
    smiles = matchms.utils.convert_inchi_to_smiles(inchi) # OLD matchms syntax
    #smiles = matchms.metadata_utils.convert_inchi_to_smiles(inchi) # NEW matchms syntax
    print(inchi)
    print(smiles)
    # Eval Pending on whether the try except is necessary here.
    try:
        smiles = matchms.utils.mol_converter(inchi, "inchi", "smiles") # OLD matchms syntax
        #smiles = matchms.metadata_utils.mol_converter(inchi, "inchi", "smiles") # NEW matchms syntax
        #smiles = matchms.metadata_utils.convert_inchi_to_smiles(inchi) # NEW matchms syntax
        smiles = smiles.strip(' ')
    except:
        smiles = ''
    # Get classyfire results
    safe_smiles = urllib.parse.quote(smiles)  # url encoding
    try:
        cf_result = get_cf_classes(safe_smiles, inchi)
    except:
        cf_result = None
    if not cf_result:
        cf_result = [None for _ in range(5)]
    npc_result = get_npc_classes(safe_smiles)
    # Get npc
    try:
        npc_result = get_npc_classes(safe_smiles)
    except:
        npc_result = None
    if not npc_result:
        npc_result = [None for _ in range(4)]
    output = {
        'inchi': inchi, 'smiles':safe_smiles,
        'cf_kingdom': cf_result[0], 
        'cf_superclass': cf_result[1], 
        'cf_class': cf_result[2], 
        'cf_subclass': cf_result[3], 
        'cf_direct_parent': cf_result[4], 
        'npc_class' : npc_result[0],
        'npc_superclass' : npc_result[1],
        'npc_pathway' : npc_result[2],
        'npc_isglycoside' : npc_result[3]}
    return output

def do_url_request(url: str) -> [bytes, None]:
    """
    Do url request and return bytes from .read() or None if HTTPError is raised
    :param url: url to access
    :return: open file or None if request failed
    """
    time.sleep(1)  # to not overload the api
    try:
        with urllib.request.urlopen(url) as inf:
            result = inf.read()
    except (urllib.error.HTTPError, urllib.error.URLError):
        # apparently the request failed
        result = None
    return result

def get_json_cf_results(raw_json: bytes) -> List[str]:
    """
    Extract the wanted CF classes from bytes version (open file) of json str
    Names of the keys extracted in order are:
    'kingdom', 'superclass', 'class', 'subclass', 'direct_parent'
    List elements are concatonated with '; '.
    :param raw_json: Json str as a bytes object containing ClassyFire
        information
    :return: Extracted CF classes
    """
    wanted_info = []
    cf_json = json.loads(raw_json)
    wanted_keys_list_name = ['kingdom', 'superclass', 'class',
                             'subclass', 'direct_parent']
    for key in wanted_keys_list_name:
        info_dict = cf_json.get(key, "")
        info = ""
        if info_dict:
            info = info_dict.get('name', "")
        wanted_info.append(info)
    return wanted_info

def get_json_npc_results(raw_json: bytes) -> List[str]:
    """Read bytes version of json str, extract the keys in order
    Names of the keys extracted in order are:
    class_results, superclass_results, pathway_results, isglycoside.
    List elements are concatonated with '; '.
    :param raw_json:Json str as a bytes object containing NPClassifier
        information
    :return: Extracted NPClassifier classes
    """
    wanted_info = []
    cf_json = json.loads(raw_json)
    wanted_keys_list = ["class_results", "superclass_results",
                        "pathway_results"]
    # this one returns a bool not a list like the others
    last_key = "isglycoside"
    for key in wanted_keys_list:
        info_list = cf_json.get(key, "")
        info = ""
        if info_list:
            info = "; ".join(info_list)
        wanted_info.append(info)
    last_info_bool = cf_json.get(last_key, "")
    last_info = "0"
    if last_info_bool:
        last_info = "1"
    wanted_info.append(last_info)

    return wanted_info

def get_cf_classes(smiles: str, inchi: str) -> Union[None, List[str]]:
    """ Get ClassyFire classes through GNPS API.

    :param smiles: Smiles for the query spectrum
    :param inchi: Inchikey for the query spectrum

    :return: List of strings with ClassyFire classes if provided by GNPS api ['cf_kingdom' 'cf_superclass' 'cf_class' 
        'cf_subclass' 'cf_direct_parent'], or None. 
    """
    classes_list = None
    if smiles is not None:
        cf_url_base_smiles = "https://gnps-structure.ucsd.edu/classyfire?smiles="
        cf_url_query_smiles = cf_url_base_smiles + smiles
        smiles_query_result = do_url_request(cf_url_query_smiles)
        if smiles_query_result is not None:
            classes_list = get_json_cf_results(smiles_query_result)
    if not classes_list: # do only if smiles query not successful.
        if inchi is not None:
            cf_url_query_inchi = f"https://gnps-classyfire.ucsd.edu/entities/{inchi}.json"
            inchi_query_result = do_url_request(cf_url_query_inchi)
            if inchi_query_result is not None:
                classes_list = get_json_cf_results(inchi_query_result)
    return classes_list

# DONE. NOT PURE FUNCTION! ATTEMPTS CONNECTION TO REMOTE (output depends on remote state).
def get_npc_classes(smiles: str) -> Union[None, List[str]]:
    """ Get NPClassifier classes through GNPS API.

    :param smiles: Smiles for the query spectrum
    :return: List of strings with NPClassifier classes if provided by GNPS api ['npc_class' 'npc_superclass' 
        'npc_pathway' 'npc_isglycoside'], or None. 
    """
    classes_list = None
    if smiles is not None:
        npc_url_base_smiles = "https://npclassifier.ucsd.edu/classify?smiles="
        npc_url_query_smiles = npc_url_base_smiles + smiles
        query_result_json = do_url_request(npc_url_query_smiles)
        if query_result_json is not None:
            classes_list = get_json_npc_results(query_result_json)
    return classes_list

# DONE. PURE FUNCTION.
def _return_model_filepath(path : str, model_suffix:str) -> str:
    """ Function parses path input into a model filepath. If a model filepath is provided, it is returned unaltered , if 
    a directory path is provided, the model filepath is searched for and returned.
    
    :param path: File path or directory containing model file with provided model_suffix.
    :param model_suffix: Model file suffix (str)
    :returns: Filepath (str).
    :raises: Error if no model in file directory or filepath does not exist. Error if more than one model in directory.
    """
    filepath = []
    if path.endswith(model_suffix):
        # path provided is a model file, use the provided path
        filepath = path
        assert os.path.exists(filepath), "Provided filepath does not exist!"
    else:
        # path provided is not a model filepath, search for model file in provided directory
        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith(model_suffix):
                    filepath.append(os.path.join(root, file))
        assert filepath is not [], f"No model file found in given path with suffix '{model_suffix}'!"
        assert len(filepath) == 1, (
        "More than one possible model file detected in directory! Please provide non-ambiguous model directory or"
        "filepath!")
    return filepath

# DONE. PURE FUNCTION.
def compute_similarities_ms2ds(spectrum_list:List[matchms.Spectrum], model_path:str) -> np.ndarray:
    """ Function computes pairwise similarity matrix for list of spectra using pretrained ms2deepscore model.
    
    :param spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
    :param model_path: Location of ms2deepscore pretrained model file path (filename ending in .hdf5 or file-directory)
    :returns: ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
    """
    filename = _return_model_filepath(model_path, ".hdf5")
    model = load_model(filename) # Load ms2ds model
    similarity_measure = MS2DeepScore(model)
    scores_matchms = calculate_scores(spectrum_list, spectrum_list, similarity_measure, is_symmetric=True)
    scores_ndarray = scores_matchms.scores
    return scores_ndarray

# DONE. PURE FUNCTION.
def compute_similarities_s2v(spectrum_list:List[matchms.Spectrum], model_path:str) -> np.ndarray:
    """ Function computes pairwise similarity matrix for list of spectra using pretrained spec2vec model.
    
    :param spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
    :param model_path: Location of spec2vec pretrained model file path (filename ending in .model or file-directory)
    :returns: ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
    """
    filename = _return_model_filepath(model_path, ".model")
    model = gensim.models.Word2Vec.load(filename) # Load s2v model
    similarity_measure = Spec2Vec(model=model)
    scores_matchms = calculate_scores(spectrum_list, spectrum_list, similarity_measure, is_symmetric=True)
    scores_ndarray = scores_matchms.scores
    return scores_ndarray

# DONE. PURE FUNCTION.
def compute_similarities_cosine(spectrum_list:List[matchms.Spectrum], cosine_type : str = "ModifiedCosine"):
    """ Function computes pairwise similarity matrix for list of spectra using specified cosine score. 
    
    :param spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
    :param cosine_type: String identifier of supported cosine metric, options: ["ModifiedCosine", "CosineHungarian", 
        "CosineGreedy"]
    :returns: ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
    """
    valid_types = ["ModifiedCosine", "CosineHungarian", "CosineGreedy"]
    assert cosine_type in valid_types, f"Cosine type specification invalid. Use one of: {str(valid_types)}"
    if cosine_type == "ModifiedCosine":
        similarity_measure = ModifiedCosine()
    elif cosine_type == "CosineHungarian":
        similarity_measure = CosineHungarian()
    elif cosine_type == "CosineGreedy":
        similarity_measure = CosineGreedy()
    tmp = calculate_scores(spectrum_list, spectrum_list, similarity_measure, is_symmetric=True)
    scores = extract_similarity_scores_from_matchms_cosine_array(tmp.scores)
    return scores

# DONE. PURE FUNCTION.
def compose_function(*func): 
    """ Generic function composer making use of functools reduce. 
    
    :param *func: Any number n of input functions to be composed.
    :returns: A new function object.
    """
    def compose(f, g):
        return lambda x : f(g(x))   
    return reduce(compose, func, lambda x : x)

# DONE. PURE FUNCTION.
def harmonize_and_clean_spectrum(spectrum : matchms.Spectrum):
    """ Function harmonizes and cleans spectrum object.
    
    :param spectrum: A single matchms spectrum object. 
    :returns: A new cleaned matchms spectrum object.
    """
    processed_spectrum = copy.deepcopy(spectrum)
    # Create partial functions for compose_function()
    select_by_mz_custom = partial(select_by_mz, mz_from = 0, mz_to = 1000)
    reduce_to_number_of_peaks_custom = partial(reduce_to_number_of_peaks, n_required = 2, n_max = 200)
    # Create pipeline function through compositions
    apply_pipeline = compose_function(
        default_filters, normalize_intensities, repair_inchi_inchikey_smiles, derive_inchi_from_smiles, 
        derive_inchikey_from_inchi, harmonize_undefined_smiles, harmonize_undefined_inchi, harmonize_undefined_inchikey,
        select_by_mz_custom, reduce_to_number_of_peaks_custom)
    # Apply pipeline
    processed_spectrum = apply_pipeline(processed_spectrum)
    return processed_spectrum

# DONE. PURE FUNCTION.
def clean_spectra(input_spectrums : List[matchms.Spectrum]):
    """ Function harmonizes and cleans spectrum object ()
    
    :param spectrum: A single matchms spectrum object.
    :returns: A single matchms spectrum object.

    note:: pure function.
    """
    spectrums = copy.deepcopy(input_spectrums) 
    spectrums = [harmonize_and_clean_spectrum(s) for s in spectrums]
    return spectrums

# DONE. PURE FUNCTION.
def extract_similarity_scores_from_matchms_cosine_array(tuple_array : np.ndarray) -> np.ndarray:
    """ Function extracts similarity matrix from matchms cosine scores array.
    
    The cosine score similarity output of matchms stores output in a numpy array of pair-tuples, where each tuple 
    contains (sim, n_frag_overlap). This function extracts the sim scores, and returns a numpy array corresponding to 
    pairwise similarity matrix.

    :param tuple_array: A single matchms spectrum object.
    :returns: A np.ndarray with shape (n, n) where n is the number of spectra deduced from the dimensions of the input
    array. Each element of the ndarray contains the pairwise similarity value.
    """
    sim_data = [ ]
    for row in tuple_array:
        for elem in row:
            sim_data.append(float(elem[0])) # <- TODO: check float conversion necessary?
    return(np.array(sim_data).reshape(tuple_array.shape[0], tuple_array.shape[1]))