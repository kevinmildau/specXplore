from dataclasses import dataclass, field
import pickle
import pandas as pd
import numpy as np
from specxplore.spectrum import Spectrum
from typing import List

@dataclass
class SpecxploreSessionData ():
    """ 
    Class interface for specXplore session data containing all infor needed by the specXplore dashboard. 
    """
    spectra_specxplore : List[Spectrum]
    tsne_coordinates_table : pd.DataFrame
    classification_table : pd.DataFrame
    metadata_table : pd.DataFrame
    primary_score : np.ndarray
    secondary_score : np.ndarray
    tertiary_score : np.ndarray
    score_names : List[str]

    # Data Structures    
    def __post_init__ (self):
        """ Functions applies some basic checks on initiated session data to see whether provided data satisfies
        frontend assumptions. """

        # assert iloc overlap across all features
        # assert each iloc has a corresponding feature_id
        # assert feature_ids unique
        # TODO: determine additional critical asserts.
        return None
    
    
    # constructor
    # getters
    var = ...
    def fun(input):
        ...


def load_specxplore_object_from_pickle(filepath : str) -> SpecxploreSessionData:
    """ Function loads specXplore object from pickle and checks type validity. """
    with open(filepath, 'rb') as file:
        specxplore_object = pickle.load(file) 
    assert isinstance(specxplore_object, SpecxploreSessionData), (
        'Provided data must be a SpecxploreSessionData object!'
    )
    return specxplore_object
