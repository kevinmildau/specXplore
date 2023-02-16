from typing import List, Union, Dict
import matchms.utils # UPDATE ALERT: this may not be the same in matchms >0.14
import json
import urllib
import time
from matchms.typing import SpectrumType
import os
import gensim
from spec2vec import Spec2Vec
import matchms
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
import pandas as pd

from collections import namedtuple

_ClassificationEntry = namedtuple(
    "ClassificationEntry", 
    field_names=['inchi', 'smiles', 'cf_kingdom', 'cf_superclass', 'cf_class', 'cf_subclass', 'cf_direct_parent', 
                 'npc_class', 'npc_superclass', 'npc_pathway', 'npc_isglycoside'],
    defaults = ["Not Available" for _ in range(0, 11)])

class ClassificationEntry(_ClassificationEntry):
    """ 
    Tuple container for classification entries. 

    Parameters:
        inchi: Compound inchi string.
        smiles: Compound smiles str
        cf_kingdom: ClassyFire kingdom classification.
        cf_superclass: ClassyFire superclass classification.
        cf_class: ClassyFire class classification.
        cf_subclass: ClassyFire subclass classification.
        cf_direct_parent: ClassyFire direct_parent classification.
        npc_class: NPClassifier class classification.
        npc_superclass: NPClassifier superclass classification.
        npc_pathway: NPClassifier pathway classification.
        npc_isglycoside: NPClassifier isglycoside classification.
    """
    _slots_ = ()

def batch_run_get_classes(spectrum_list : List[matchms.Spectrum], verbose : bool = True) -> pd.DataFrame:
    """ 
    Function queries GNPS API for classes for all spectra with inchi in spectrum list.
    
    Parameters:
        spectrum_list: List of matchms spectra. These are expected to have an inchi entry available.
        verbose: Boolean indicator that controls progress prints. Default is true. Deactive prints by setting to False.
    Returns:
        A pandas.DataFrame constructed from ClassificationEntry tuples. In addition, the list index is added as 
        "iloc_spectrum_identifier" column.
    """
    classes_list = []
    for iloc, spectrum in enumerate(spectrum_list):
        if verbose and (iloc+1) % 10 == 0 and not iloc == 0:
            print(f"{iloc + 1} spectra done, {len(spectrum_list) - (iloc+1)} spectra remaining.")
        inchi = spectrum.get("inchi")
        classes_list.append(get_classes(inchi))
    classes_df = pd.DataFrame.from_records(classes_list, columns=ClassificationEntry._fields)
    classes_df["iloc_spectrum_identifier"] = classes_df.index
    return classes_df

def get_classes(inchi: Union[str, None]) -> ClassificationEntry:
    """
    Function returns cf (classyfire) and npc (natural product classifier) classes for a provided inchi.
    
    Parameters
        inchi: A valid inchi for which class information should be fetched. An input of "" or None is handled as an 
               exception with a dict of "Not Available" data being returned.
    Returns:
        ClassificationEntry named tuple with classification information if available. If classification retrieval fails,
        ClassificationEntry will contain "Not Available" defaults.
    """
    if inchi is None or inchi == "":
        print("No inchi, returning Not Available structure.")
        return ClassificationEntry()
    smiles = matchms.utils.convert_inchi_to_smiles(inchi) # OLD matchms syntax
    #smiles = matchms.metadata_utils.convert_inchi_to_smiles(inchi) # NEW matchms syntax
    # Get ClassyFire classifications
    safe_smiles = urllib.parse.quote(smiles)  # url encoding
    try:
        cf_result = get_cf_classes(safe_smiles, inchi)
    except:
        cf_result = None
    if not cf_result:
        cf_result = [None for _ in range(5)]
    npc_result = get_npc_classes(safe_smiles)
    # Get NPClassifier classifications
    try:
        npc_result = get_npc_classes(safe_smiles)
    except:
        npc_result = None
    if not npc_result:
        npc_result = [None for _ in range(4)]
    output = ClassificationEntry(
        inchi= inchi, smiles=safe_smiles, cf_kingdom=cf_result[0], cf_superclass=cf_result[1], cf_class=cf_result[2],
        cf_subclass=cf_result[3], cf_direct_parent=cf_result[4], npc_class=npc_result[0], npc_superclass=npc_result[1],
        npc_pathway=npc_result[2], npc_isglycoside=npc_result[3])
    return output

def do_url_request(url: str, sleep_time_seconds : int = 1) -> Union[bytes, None]:
    """ 
    Perform url request and return bytes from .read() or None if HTTPError is raised.

    Parameters:
        url: url string that should be accessed.
        sleep_time_seconds: integer value indicating the number of seconds to wait in between API requests.
    :param url: url to access
    :return: open file or None if request failed
    """
    time.sleep(sleep_time_seconds) # Added to prevent API overloading.
    try:
        with urllib.request.urlopen(url) as inf:
            result = inf.read()
    except (urllib.error.HTTPError, urllib.error.URLError): # request fail => None result
        result = None
    return result


def get_json_cf_results(raw_json: bytes) -> List[str]:
    """ 
    Extracts ClassyFire classification key data in order from bytes version of json string.
    
    Names of the keys extracted in order are: ['kingdom', 'superclass', 'class', 'subclass', 'direct_parent']
    List elements are concatenated with '; '.

    :param raw_json: Json str as a bytes object containing ClassyFire information
    :return:List of extracted ClassyFire class assignment strings.
    """
    cf_results_list = []
    json_string = json.loads(raw_json)
    key_list = ['kingdom', 'superclass', 'class', 'subclass', 'direct_parent']
    for key in key_list:
        data_dict = json_string.get(key, "")
        data_string = ""
        if data_dict:
            data_string = data_dict.get('name', "")
        cf_results_list.append(data_string)
    return cf_results_list


def get_json_npc_results(raw_json: bytes) -> List[str]:
    """ Extracts NPClassifier classification key data in order from bytes version of json string.
    
    Names of the keys extracted in order are: class_results, superclass_results, pathway_results, isglycoside.
    List elements are concatenated with '; '.

    :param raw_json: Json str as a bytes object containing NPClassifier information.
    :return: List of extracted NPClassifier class assignment strings.
    """
    npc_results_list = []
    json_string = json.loads(raw_json)
    # Key list extraction
    key_list = ["class_results", "superclass_results", "pathway_results"]
    for key in key_list:
        data_list = json_string.get(key, "")
        data_string = ""
        if data_list:
            data_string = "; ".join(data_list)
        npc_results_list.append(data_string)
    # Boolean key special extraction
    last_key = "isglycoside" # requires special treatment since boolean
    data_last = json_string.get(last_key, "")
    last_string = "0"
    if data_last:
        last_string = "1"
    npc_results_list.append(last_string)
    return npc_results_list

# DONE.  NOT PURE FUNCTION! ATTEMPTS CONNECTION TO REMOTE (output depends on remote state).
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
    if classes_list is not None: # do only if smiles query not successful.
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
        assert len(filepath) > 0, f"No model file found in given path with suffix '{model_suffix}'!"
        assert len(filepath) == 1, (
        "More than one possible model file detected in directory! Please provide non-ambiguous model directory or"
        "filepath!")
    return filepath[0]

def compute_similarities_ms2ds(spectrum_list:List[matchms.Spectrum], model_path:str) -> np.ndarray:
    """ Function computes pairwise similarity matrix for list of spectra using pretrained ms2deepscore model.
    
    Parameters
        spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
        model_path: Location of ms2deepscore pretrained model file path (filename ending in .hdf5 or file-directory)
    Returns: 
        ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
    """
    filename = _return_model_filepath(model_path, ".hdf5")
    model = load_model(filename) # Load ms2ds model
    similarity_measure = MS2DeepScore(model)
    scores_matchms = calculate_scores(spectrum_list, spectrum_list, similarity_measure, is_symmetric=True)
    scores_ndarray = scores_matchms.scores
    return scores_ndarray


def compute_similarities_s2v(spectrum_list:List[matchms.Spectrum], model_path:str) -> np.ndarray:
    """ Function computes pairwise similarity matrix for list of spectra using pretrained spec2vec model.
    
    Parameters:
        spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
        model_path: Location of spec2vec pretrained model file path (filename ending in .model or file-directory)
    Returns: 
        ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
    """
    filename = _return_model_filepath(model_path, ".model")
    model = gensim.models.Word2Vec.load(filename) # Load s2v model
    similarity_measure = Spec2Vec(model=model)
    scores_matchms = calculate_scores(spectrum_list, spectrum_list, similarity_measure, is_symmetric=True)
    scores_ndarray = scores_matchms.scores
    return scores_ndarray


def compute_similarities_cosine(spectrum_list:List[matchms.Spectrum], cosine_type : str = "ModifiedCosine"):
    """ Function computes pairwise similarity matrix for list of spectra using specified cosine score. 
    
    Parameters:
        spectrum_list: List of matchms ms/ms spectra. These should be pre-processed and must incldue peaks.
        cosine_type: String identifier of supported cosine metric, options: ["ModifiedCosine", "CosineHungarian", 
        "CosineGreedy"]
    Returns:
        ndarray with shape (n, n) where n is the number of spectra (Pairwise similarity matrix).
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


def compose_function(*func) -> object: 
    """ Generic function composer making use of functools reduce. 
    
    Parameters:
        *func: Any number n of input functions to be composed.
    Returns: 
        A new function object.
    """
    def compose(f, g):
        return lambda x : f(g(x))   
    return reduce(compose, func, lambda x : x)


def harmonize_and_clean_spectrum(spectrum : matchms.Spectrum):
    """ Function harmonizes and cleans spectrum object.
    
    Parameters:
        spectrum: A single matchms spectrum object. 
    Returns: 
        A new cleaned matchms spectrum object.
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


def clean_spectra(input_spectrums : List[matchms.Spectrum]):
    """ 
    Function harmonizes and cleans spectrum object ()
    
    Parameters:
        spectrum: A single matchms spectrum object.
    Returns: 
        A single matchms spectrum object.
    """
    spectrums = copy.deepcopy(input_spectrums) 
    spectrums = [harmonize_and_clean_spectrum(s) for s in spectrums]
    return spectrums


def extract_similarity_scores_from_matchms_cosine_array(tuple_array : np.ndarray) -> np.ndarray:
    """ 
    Function extracts similarity matrix from matchms cosine scores array.
    
    The cosine score similarity output of matchms stores output in a numpy array of pair-tuples, where each tuple 
    contains (sim, n_frag_overlap). This function extracts the sim scores, and returns a numpy array corresponding to 
    pairwise similarity matrix.

    Parameters:
        tuple_array: A single matchms spectrum object.
    Returns: 
        A np.ndarray with shape (n, n) where n is the number of spectra deduced from the dimensions of the input
        array. Each element of the ndarray contains the pairwise similarity value.
    """
    sim_data = [ ]
    for row in tuple_array:
        for elem in row:
            sim_data.append(float(elem[0])) # <- TODO: check float conversion necessary?
    return(np.array(sim_data).reshape(tuple_array.shape[0], tuple_array.shape[1]))

def convert_similarity_to_distance(similarity_matrix : np.ndarray) -> np.ndarray:
    """ 
    Converts pairwise similarity matrix to distance matrix with values between 0 and 1 
    """
    distance_matrix = 1.- similarity_matrix
    distance_matrix = np.round(distance_matrix, 6) # Round to deal with floating point issues
    distance_matrix = np.clip(distance_matrix, a_min = 0, a_max = 1) # Clip to deal with floating point issues
    return distance_matrix