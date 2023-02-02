import pandas as pd
from typing import List, Tuple, Union
import matchms.utils
import json
import urllib
import time
from sys import argv
from typing import List, Union, Dict
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
from matchms.filtering import default_filters
from matchms.filtering import repair_inchi_inchikey_smiles
from matchms.filtering import derive_inchikey_from_inchi
from matchms.filtering import derive_smiles_from_inchi
from matchms.filtering import derive_inchi_from_smiles
from matchms.filtering import harmonize_undefined_inchi
from matchms.filtering import harmonize_undefined_inchikey
from matchms.filtering import harmonize_undefined_smiles
from matchms.filtering import normalize_intensities
from matchms.filtering import select_by_mz
from matchms.filtering import reduce_to_number_of_peaks

def get_classes(inchi: str):
    """Function returns cf (classyfire) and npc (natural product classifier) 
    classes for a provided inchi.
    
    Sould no information be available, the output will be a list of 
    "Not Available" strings of length 9 for 5 cf classes and 4 npc classes for
    downstream conversion to pandas data frame compatibility.

    Context: Effectively fetches and constructs single row of pandas df using
    a list of dictionaries.

    Args
    ------
    inchi
        A valid inchi for which class information should be fetched. An 
        input of "" is handles as an exception with a dict of "Not Available"
        data being returned.
    """
    key_list = ['inchi_key', 'smiles', 'cf_kingdom', 'cf_superclass',
        'cf_class', 'cf_subclass', 'cf_direct_parent', 'npc_class', 
        'npc_superclass', 'npc_pathway', 'npc_isglycoside']
    if inchi == "":
        print("No inchi, returning Not Available structure.")
        output = {key:"Not Available" for key in key_list}
        return output
    
    smiles = matchms.utils.convert_inchi_to_smiles(inchi)
    #smiles = matchms.metadata_utils.convert_inchi_to_smiles(inchi)
    print(inchi)
    print(smiles)
    # Eval Pending on whether the try except is necessary here.
    try:
        smiles = matchms.utils.mol_converter(inchi, "inchi", "smiles")
        #smiles = matchms.metadata_utils.mol_converter(inchi, "inchi", "smiles")
        #smiles = matchms.metadata_utils.convert_inchi_to_smiles(inchi)
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
    
    output = {'inchi': inchi, 'smiles':safe_smiles,
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
    """Get ClassyFire classes through GNPS API
    :param smiles: Smiles for the query spectrum
    :param inchi: Inchikey for the query spectrum
    :return: ClassyFire classes if possible
    """
    result = None
    # lookup CF with smiles
    if smiles:
        url_base = "https://gnps-structure.ucsd.edu/classyfire?smiles="
        url_smiles = url_base + smiles
        smiles_result = do_url_request(url_smiles)

        # read CF result
        if smiles_result is not None:
            result = get_json_cf_results(smiles_result)

    if not result:
        # do a second try with inchikey
        if inchi:
            url_inchi = \
                f"https://gnps-classyfire.ucsd.edu/entities/{inchi}.json"
            inchi_result = do_url_request(url_inchi)

            # read CF result from inchikey lookup
            if inchi_result is not None:
                result = get_json_cf_results(inchi_result)
    return result


def get_npc_classes(smiles: str) -> Union[None, List[str]]:
    """Get NPClassifier classes through GNPS API
    :param smiles: Smiles for the query spectrum
    :return: NPClassifier classes if possible
    """
    result = None
    # lookup NPClassifier with smiles
    if smiles:
        url_base_npc = "https://npclassifier.ucsd.edu/classify?smiles="
        url_smiles_npc = url_base_npc + smiles
        smiles_result_npc = do_url_request(url_smiles_npc)

        # read NPC result
        if smiles_result_npc is not None:
            result = get_json_npc_results(smiles_result_npc)
    return result



def _return_model_filepath(path, model_suffix):
    """ Helper function parses given path and checks whether it is a model
    file path (based of suffix) or directory. If the former, returns the path, 
    if the latter, searches for file with given suffix and returns its
    filepath."""
    filename = None
    if path.endswith(model_suffix):
        # path provided is a model file, use the provided model file
        filename = path
    else:
        # path provided is not a model file, search for model file in path
        for root, dirs, files in os.walk(path):
            for file in files:
                if file.endswith(model_suffix):
                    filename = os.path.join(root, file)
    # Find ms2deepscore trained model in provided filepath.
    assert filename is not None, f"No model file found in given path with suffix '{model_suffix}'"
    return filename

def compute_similarities_ms2ds(spectrum_list, path):
    """ Function computes pairwise similarity matrix for given spectrum list 
    using pretrained ms2deepscore model provided with path (filename ending in 
    .hdf5 or file-directory) """
    
    # Find / set model file location
    filename = _return_model_filepath(path, ".hdf5")

    # Load model
    model = load_model(filename)

    
    # Calculate scores and get matchms.Scores object
    similarity_measure = MS2DeepScore(model)
    scores = calculate_scores(
        spectrum_list, spectrum_list, similarity_measure, is_symmetric=True)
    # Run comparison
    # Retrun pairwise similarity matrix
    return scores.scores

def compute_similarities_s2v(spectrum_list, path):
    """ Function computes pairwise similarity matrix for given spectrum list 
    using pretrained spec2vec model provided with path (filename ending in 
    .model or file-directory) """
    
    # Find / set model file location
    filename = _return_model_filepath(path, ".model")

    # Load model
    model = gensim.models.Word2Vec.load(filename)
    spec2vec_similarity = Spec2Vec(model=model)

    # Run comparison
    scores = calculate_scores(
        spectrum_list, spectrum_list, spec2vec_similarity, is_symmetric=True)
    # Retrun pairwise similarity matrix
    return scores.scores # extract ndarray from matchms.scores object

def compute_similarities_cosine(spectrum_list, type = "ModifiedCosine"):
    """ Function computes pairwise similarity matrix for given spectrum list 
    using modified cosine scores """

    valid_types = ["ModifiedCosine", "CosineHungarian", "CosineGreedy"]
    assert type in valid_types, f"Cosine type specification invalid. Use one of: {str(valid_types)}"
    if type == "ModifiedCosine":
        similarity_measure = ModifiedCosine()
    elif type == "CosineHungarian":
        similarity_measure = CosineHungarian()
    elif type == "CosineGreedy":
        similarity_measure = CosineGreedy()

    # Run comparison
    tmp = calculate_scores(
        spectrum_list, spectrum_list, similarity_measure, is_symmetric=True)
    scores = extract_similarity_scores(tmp.scores)
    # Retrun pairwise similarity matrix
    return scores # extract ndarray from matchms.scores object



def data_processing(spectrum):
    """ Metadata filter pipeline. """
    spectrum = default_filters(spectrum)
    spectrum = normalize_intensities(spectrum)
    spectrum = repair_inchi_inchikey_smiles(spectrum)
    spectrum = derive_inchi_from_smiles(spectrum)
    spectrum = derive_smiles_from_inchi(spectrum)
    spectrum = derive_inchikey_from_inchi(spectrum)
    spectrum = harmonize_undefined_smiles(spectrum)
    spectrum = harmonize_undefined_inchi(spectrum)
    spectrum = harmonize_undefined_inchikey(spectrum)
    spectrum = select_by_mz(spectrum, 0, 1000)
    spectrum = reduce_to_number_of_peaks(spectrum, n_required = 2, n_max = 200)
    return spectrum

def clean_spectra(input_spectrums):
    # deepcopy avoids any modification of original spectrums unless explicitly
    # done by user outside of function.
    spectrums = copy.deepcopy(input_spectrums) 
    spectrums = [data_processing(s) for s in spectrums]
    return spectrums



def extract_similarity_scores(sm):
    """Function extracts similarity matrix from matchms cosine scores array.
      The cosine scores similarity output of matchms stores output in a
      numpy array of pair-tuples, where each tuple contains 
      (sim, n_frag_overlap). This function extracts the sim scores, and returns
      a numpy array corresponding to pairwise similarity matrix.
    """
    sim_data = [ ]
    for row in sm:
        for elem in row:
            sim_data.append(float(elem[0]))
    return(np.array(sim_data).reshape(sm.shape[0],sm.shape[1]))