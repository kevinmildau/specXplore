import matchms.utils 
import json
import urllib
import time
import os
import gensim
from spec2vec import Spec2Vec
import matchms
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine, CosineHungarian
#from ms2query.utils import load_matchms_spectrum_objects_from_file
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
import copy
import numpy as np
import pandas as pd
from collections import namedtuple
from networkx import read_graphml
from networkx.readwrite import json_graph
from typing import List, Union

# Named Tuple Basis for ClassificationEntry class
_ClassificationEntry = namedtuple(
    "ClassificationEntry", 
    field_names=[
        'inchi', 'smiles', 'cf_kingdom', 'cf_superclass', 'cf_class', 'cf_subclass', 'cf_direct_parent', 
        'npc_class', 'npc_superclass', 'npc_pathway', 'npc_isglycoside'],
    defaults = ["Not Available" for _ in range(0, 11)])

class ClassificationEntry(_ClassificationEntry):
    """ 
    Tuple container class for classification entries. 

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




def initialize_classification_output_file(filepath) -> None:
    """ Creates csv file with ClassificationEntry headers if not exists at filepath. """
    # Initialize the file
    if not os.path.isfile(filepath):
        with open(filepath, "w") as file:
            pass
    # Add header line to file
    if os.stat(filepath).st_size == 0:
        with open(filepath, "a") as file: # a for append mode
            file.write(", ".join(ClassificationEntry._fields) + os.linesep)
    return None

def append_classes_to_file(classes : ClassificationEntry, filepath : str) -> None:
    """ Appends classification entry data to file. """
    pandas_row = pd.DataFrame([classes])
    pandas_row.to_csv(filepath, mode='a', header=False, sep = ",", na_rep="Not Available", index = False)
    return None

def batch_run_get_classes(
        inchi_list : List[str], 
        filepath : str, 
        verbose : bool = True) -> pd.DataFrame:
    """ 
    Function queries GNPS API for NPClassifier and ClassyFire classifications for all inchi list. 
    
    A pandas.DataFrame is returned as output, and the corresponding csv is saved to file iteratively. This is done to
    allow run continuation in case of API disconnect errors.
    
    Parameters:
        inchi_list: List of inchi strings.
        filename: str file path for output to be saved to iteratively.
        verbose: Boolean indicator that controls progress prints. Default is true. Deactive prints by setting to False.
    
    Returns:
        A pandas.DataFrame constructed from ClassificationEntry tuples. In addition, the list index is added as 
        "iloc_spectrum_identifier" column.

        Also saves intermediate results to csv file.
    """
    classes_list = []
    initialize_classification_output_file(filepath)
    for iloc, inchi in enumerate(inchi_list):
        if verbose and (iloc+1) % 10 == 0 and not iloc == 0:
            print(f"{iloc + 1} spectra done, {len(inchi_list) - (iloc+1)} spectra remaining.")
        classes = get_classes(inchi)
        append_classes_to_file(classes, filepath)
        classes_list.append(classes)
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
        ClassificationEntry will contain "Not Available" defaults. "Not Available" defaults may also be produced by
        server disconnections while in principle the classification may be obtainable.
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
        cf_result = ["Not Available" for _ in range(5)]

    # Get NPClassifier classifications
    try:
        npc_result = get_npc_classes(safe_smiles)
    except:
        npc_result = None
    if not npc_result:
        npc_result = ["Not Available" for _ in range(4)]
        
    output = ClassificationEntry(
        inchi= inchi, smiles=safe_smiles, cf_kingdom=cf_result[0], cf_superclass=cf_result[1], cf_class=cf_result[2],
        cf_subclass=cf_result[3], cf_direct_parent=cf_result[4], npc_class=npc_result[0], npc_superclass=npc_result[1],
        npc_pathway=npc_result[2], npc_isglycoside=npc_result[3])
    return output

def do_url_request(url: str, sleep_time_seconds : int = 2) -> Union[bytes, None]:
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

def read_list_from_text(filename : str) -> List[str]:
    """ Reads newline separated file into list. """
    with open(filename) as f:
        output_list = f.read().splitlines()
    return output_list

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
            sim_data.append(float(elem[0]))
    return(np.array(sim_data).reshape(tuple_array.shape[0], tuple_array.shape[1]))


def extract_molecular_family_assignment_from_graphml(filepath : str) -> pd.DataFrame:
    """ Function extracts molecular family componentindex for each node in gnps mgf export. Expects that each
    spectrum is a feature, hence the clustering option in molecular networking must be deactivated. """
    graph = read_graphml(filepath)
    data = json_graph.node_link_data(graph)
    entries = []
    for node in data['nodes']:
        entry = {"id" : node['id'], "spectrum_id" : node['SpectrumID'], 'molecular_family' : node['componentindex']}
        entries.append(entry)
    df = pd.DataFrame.from_records(entries)
    df['id'] = df['id'].astype(int)
    df['idx'] = df['id'] -1
    df.sort_values(by = "id", inplace=True)
    df.reset_index(drop = True, inplace=True)
    return df

def apply_basic_matchms_filters_to_spectra(
        input_spectra : List[matchms.Spectrum],
        minimum_number_of_peaks_per_spectrum : int = 3,
        maximum_number_of_peaks_per_spectrum : int = 200,
        max_mz = 1000,
        min_mz = 0,
        verbose = True
        ) -> List[matchms.Spectrum]:
    ''' Applies basic pre-processing of spectra required for specXplore processing.'''
    if verbose:
        print("Number of spectra prior to filtering: ", len(input_spectra))
    # Normalize intensities, important for similarity measures!
    output_spectra = copy.deepcopy(input_spectra)
    output_spectra = [matchms.filtering.normalize_intensities(spec) for spec in output_spectra]
    output_spectra = [matchms.filtering.select_by_mz(spec, mz_from = 0, mz_to = 1000) for spec in output_spectra]
    # Clean spectra by remove very low intensity fragments, noise removal
    output_spectra = [
        matchms.filtering.reduce_to_number_of_peaks(
            spec, n_required = minimum_number_of_peaks_per_spectrum, n_max= maximum_number_of_peaks_per_spectrum) 
        for spec in output_spectra]
    # Add precursor mz values to matchms spectrum entry
    output_spectra = [matchms.filtering.add_precursor_mz(spec)  for spec in output_spectra]
    # remove none entries in list (no valid spectrum returned)
    output_spectra = [spec for spec in output_spectra if spec is not None]
    if verbose:
        print("Number of spectra after to filtering: ", len(output_spectra))
    return output_spectra