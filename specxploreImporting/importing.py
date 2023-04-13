from typing import List, Union, Dict, Tuple
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
from ms2query.utils import load_matchms_spectrum_objects_from_file
from ms2query.ms2library import MS2Library
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
import copy
import numpy as np
from functools import reduce, partial
from matchms.filtering import (default_filters, repair_inchi_inchikey_smiles, derive_inchikey_from_inchi, 
    derive_smiles_from_inchi, derive_inchi_from_smiles, harmonize_undefined_inchi, harmonize_undefined_inchikey,
    harmonize_undefined_smiles, normalize_intensities, select_by_mz, reduce_to_number_of_peaks)
import pandas as pd
from kmedoids import KMedoids
from sklearn.metrics import silhouette_score
from dataclasses import dataclass
import plotly.express as px
from collections import namedtuple
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr  
from sklearn.manifold import TSNE
import plotly.graph_objects as go

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



@dataclass
class KmedoidGridEntry():
    """ 
    Container Class for K medoid clustering results.

    Parameters:
        k: the number of clusters aimed for.
        cluster_assignments: List with cluster assignment for each observation.
        silhouette_score: float with clustering silhouette score.
    """
    k : int
    cluster_assignments : List[int]
    silhouette_score : float
    random_seed_used : int
    def __str__(self) -> str:
        custom_print = (
            f"k = {self.k}, silhoutte_score = {self.silhouette_score}, \n"
            f"cluster_assignment = {', '.join(self.cluster_assignments[0:7])}...")
        return custom_print

def initialize_classification_output_file(filepath) -> None:
    """ Creates csv file with ClassificationEntry headers if not exists at filepath. """
    if not os.path.isfile(filepath):
        with open(filepath, "w") as file:
            pass
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
        filename : str, 
        verbose : bool = True) -> pd.DataFrame:
    """ 
    Function queries GNPS API for classes for all spectra with inchi in spectrum list. 
    
    A pandas.DataFrame is returned as output, and the corresponding csv is saved to file iteratively. This is done to
    allow run continuation in case of API disconnect errors.
    
    Parameters:
        spectrum_list: List of matchms spectra. These are expected to have an inchi entry available.
        filename: str file path for output to be saved to iteratively.
        verbose: Boolean indicator that controls progress prints. Default is true. Deactive prints by setting to False.
    Returns:
        A pandas.DataFrame constructed from ClassificationEntry tuples. In addition, the list index is added as 
        "iloc_spectrum_identifier" column.

        Also saves intermediate results to csv file.
    """
    classes_list = []
    initialize_classification_output_file(filename)
    for iloc, inchi in enumerate(inchi_list):
        if verbose and (iloc+1) % 10 == 0 and not iloc == 0:
            print(f"{iloc + 1} spectra done, {len(inchi_list) - (iloc+1)} spectra remaining.")
        classes = get_classes(inchi)
        append_classes_to_file(classes, filename)
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


def harmonize_and_clean_spectrum(
        spectrum : matchms.Spectrum,
        minimum_number_of_required_peaks_per_spectrum = 4,
        maximum_number_of_peaks_allowed_per_spectrum = 200,
        minimum_mz_for_fragment_in_spectrum = 0,
        maximum_mz_for_fragment_in_spectrum = 1000,
        minimum_relative_intensity_for_fragments = 0.01):
    """ Function harmonizes and cleans spectrum object.
    
    Parameters:
        spectrum: A single matchms spectrum object. 
    Returns: 
        A new cleaned matchms spectrum object.
    """
    processed_spectrum = copy.deepcopy(spectrum)
    processed_spectrum = matchms.filtering.default_filters(processed_spectrum)
    processed_spectrum = matchms.filtering.normalize_intensities(processed_spectrum)
    processed_spectrum = matchms.filtering.select_by_mz(
        processed_spectrum, 
        minimum_mz_for_fragment_in_spectrum, 
        maximum_mz_for_fragment_in_spectrum)
    processed_spectrum = matchms.filtering.select_by_relative_intensity(
        processed_spectrum, 
        intensity_from = minimum_relative_intensity_for_fragments, 
        intensity_to = 1)
    processed_spectrum = matchms.filtering.reduce_to_number_of_peaks(
        processed_spectrum, 
        n_required=minimum_number_of_required_peaks_per_spectrum, 
        n_max=maximum_number_of_peaks_allowed_per_spectrum)
    processed_spectrum = matchms.filtering.repair_inchi_inchikey_smiles(processed_spectrum)
    processed_spectrum = matchms.filtering.harmonize_undefined_inchi(processed_spectrum)
    processed_spectrum = matchms.filtering.harmonize_undefined_inchikey(processed_spectrum)
    processed_spectrum = matchms.filtering.harmonize_undefined_smiles(processed_spectrum)
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
    spectrums = [s for s in spectrums if s is not None]
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


def run_kmedoid_grid(distance_matrix : np.ndarray, k_values : List[int], random_states : Union[List, None] = None):
    """ Runs k-medoid clustering for every value in k_values. """
    if random_states is None:
        random_states = [ 0 for _ in k_values ]
    output_list = []
    for k in k_values:
        assert isinstance(k, int), (
            "k must be python int object. KMedoids module requires strict Python int object (np.int64 rejected!)")
    for idx, k in enumerate(k_values):
        cluster = KMedoids(n_clusters=k, metric='precomputed', random_state=random_states[idx], method = "fasterpam")  
        cluster_assignments = cluster.fit_predict(distance_matrix)
        cluster_assignments = ["km_" + str(elem) for elem in cluster_assignments] # string conversion
        score = silhouette_score(X = distance_matrix, labels = cluster_assignments, metric= "precomputed")
        output_list.append(KmedoidGridEntry(k, cluster_assignments, score, random_states[idx]))
    return output_list

def render_kmedoid_fitting_results_in_browser(kmedoid_list : List[KmedoidGridEntry]) -> None:
    """ Plots Silhouette Score vs k for each pair in kmedoid list. """
    scores = [x.silhouette_score for x in kmedoid_list]
    ks = [x.k for x in kmedoid_list]
    fig = px.scatter(x = ks, y = scores)
    fig.update_layout(xaxis_title="K (Number of Clusters)", yaxis_title="Silhouette Score")
    fig.show(renderer = "browser")

from dataclasses import dataclass
@dataclass
class TsneGridEntry():
    """ 
    Container Class for K medoid clustering results.

    Parameters:
        k: the number of clusters aimed for.
        cluster_assignments: List with cluster assignment for each observation.
        silhouette_score: float with clustering silhouette score.
    """
    perplexity : int
    x_coordinates : List[int]
    y_coordinates:  List[int]
    pearson_score : float
    spearman_score : float
    random_seed_used : float
    def __str__(self) -> str:
        custom_print = (
            f"Perplexity = {self.perplexity}," 
            f"Pearson Score = {self.pearson_score}, "
            f"Spearman Score = {self.spearman_score}, \n"
            f"x coordinates = {', '.join(self.x_coordinates[0:4])}...",
            f"y coordinates = {', '.join(self.y_coordinates[0:4])}...")
        return custom_print

def run_tsne_grid(distance_matrix : np.ndarray, perplexity_values : List[int], random_states : Union[List, None] = None):
    """ Runs k-medoid clustering for every value in k_values. """
    if random_states is None:
        random_states = [ 0 for _ in perplexity_values ]
    output_list = []
    for idx, perplexity in enumerate(perplexity_values):
        model = TSNE(metric="precomputed", random_state = random_states[idx], init = "random", perplexity = perplexity)
        z = model.fit_transform(distance_matrix)
        # Compute embedding quality
        dist_tsne = squareform(pdist(z, 'seuclidean'))
        spearman_score = np.array(spearmanr(distance_matrix.flat, dist_tsne.flat))[0]
        pearson_score = np.array(pearsonr(distance_matrix.flat, dist_tsne.flat))[0]
        output_list.append(TsneGridEntry(perplexity, z[:,0], z[:,1], pearson_score, spearman_score, random_states[idx]))
    return output_list



def render_tsne_fitting_results_in_browser(tsne_list : List[TsneGridEntry]) -> None:
    """ Plots Silhouette Score vs k for each pair in kmedoid list. """
    pearson_scores = [x.spearman_score for x in tsne_list]
    spearman_scores = [x.pearson_score for x in tsne_list]
    perplexities = [x.perplexity for x in tsne_list]

    trace_spearman = go.Scatter(x = perplexities, y = spearman_scores, name="spearman_score", mode = "markers")
    trace_pearson = go.Scatter(x = perplexities, y = pearson_scores, name="pearson_score", mode = "markers")
    fig = go.Figure([trace_pearson, trace_spearman])
    fig.update_layout(xaxis_title="Perplexity", yaxis_title="Score")
    fig.show(renderer = "browser")


def expand_ms2query_results_table(results_table, n_spectra):
    """ Expands ms2query output to contain NA rows for unsuccesful queries."""
    # Construct complete index for all possible query_spectrum_nr entries
    new_index = pd.Index(np.arange(1, n_spectra + 1), name="query_spectrum_nr")
    # Superimpose the new index and reset index to iloc
    out_df = results_table.set_index("query_spectrum_nr")
    out_df = out_df.reindex(new_index).reset_index()
    # add iloc index column
    out_df["source_spectrum_mgf_iloc"] = out_df.index
    return out_df


def run_single_file(
    ms2library: MS2Library,
    spectra_filename: str,
    results_filename: str,
    nr_of_analogs_to_store: int = 1,
    minimal_ms2query_score: Union[int, float] = 0.0,
    additional_metadata_columns: Tuple[str] = ("retention_time", "retention_index",),
    additional_ms2query_score_columns: List[str] = None
    ) -> None:
    """
    Runs analog library search and stores search results for all spectra in provided file in results file. 
    Note that results are stored additively to the file; rerunning code appends to the existing file.

    Args:
    ------
    ms2library:
        MS2Library object
    spectra_filename:
        Path to file containing spectra on which analog search should be run.
    results_filename:
        Path to file in which the results are stored. Should be a .csv filename.
    nr_of_top_analogs_to_store:
        The number of returned analogs that are stored.
    minimal_ms2query_score:
        The minimal ms2query metascore needed to be stored in the csv file.
        Spectra for which no analog with this minimal metascore was found,
        will not be stored in the csv file.
    additional_metadata_columns:
        Additional columns with query spectrum metadata that should be added. For instance "retention_time".
    additional_ms2query_score_columns:
        Additional columns with scores used for calculating the ms2query metascore
        Options are: "mass_similarity", "s2v_score", "ms2ds_score", "average_ms2ds_score_for_inchikey14",
        "nr_of_spectra_with_same_inchikey14*0.01", "chemical_neighbourhood_score",
        "average_tanimoto_score_for_chemical_neighbourhood_score",
        "nr_of_spectra_for_chemical_neighbourhood_score*0.01"
    set_charge_to:
        The charge of all spectra that have no charge is set to this value. This is important for precursor m/z
        calculations. It is important that for positive mode charge is set to 1 and at negative mode charge is set to -1
        for correct precursor m/z calculations.
    change_all_charges:
        If True the charge of all spectra is set to this value. If False only the spectra that do not have a specified
        charge will be changed.
    """
    # Go through spectra files in directory
    spectra = load_matchms_spectrum_objects_from_file(spectra_filename)
    ms2library.analog_search_store_in_csv(
        spectra, results_filename, None)
    return None