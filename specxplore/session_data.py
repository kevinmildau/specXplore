from dataclasses import dataclass, field
import pickle
import pandas as pd
import numpy as np
from specxplore.spectrum import Spectrum
from typing import List, Union
import specxplore.importing_cython
from specxplore.constants import SELECTED_NODES_STYLE, GENERAL_STYLE, NETVIEW_STYLE
import copy

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
    highlight_table : pd.DataFrame

    # Variables derived in post init
    targets : Union[np.ndarray, None] = None
    sources: Union[np.ndarray, None] = None
    values : Union[np.ndarray, None] = None
    class_dict : Union[dict, None] = None
    available_classes : Union[List[str], None] = None
    selected_class_data : Union[List[str], None] = None
    initial_style : Union[List[dict], None] = None

    def __post_init__ (self):
        """ Functions applies some basic checks on initiated session data to see whether provided data satisfies
        frontend assumptions. """
        self._construct_edge_arrays()
        self.initialize_specxplore_dashboard_variables()
        return None
    def _construct_edge_arrays(self) -> None:
        ''' 
        Construct numpy based sources, targets, and value edge arrays. Creates three separate
        arrays where each iloc represents a respective target and source node with certain edge weight. 
        The arrays are ordered in descending order of values, meaning that the highest similarity edges come first.
        Sources and targets are int64 corresponding to spectrum_ilocs.
        '''
        sources, targets, values = specxplore.importing_cython.construct_long_format_sim_arrays(
            self.primary_score
        )  
        ordered_index = np.argsort(-values)
        sources = sources[ordered_index]
        targets = targets[ordered_index]
        values = values[ordered_index]
        self.sources = sources
        self.targets = targets
        self.values = values
        return None
    def initialize_specxplore_dashboard_variables(self) -> None:
        ''' Construct variables derived from input that are used inside the dashboard. 
        These will be internal, private style variables, left accessible to the user however. 
        '''
        class_table = self.get_class_table()
        self.class_dict = {
            elem : list(class_table[elem]) 
            for elem in class_table.columns
        } 
        self.available_classes = list(self.class_dict.keys())
        self.selected_class_data = self.class_dict[self.available_classes[0]] # initialize default
        self.initial_style = SELECTED_NODES_STYLE + GENERAL_STYLE + NETVIEW_STYLE
        return None
    def get_class_table(self) -> pd.DataFrame:
        """ Returns class table for use within specXplore; omits spectrum_iloc and feature_id columns. """
        output_table = copy.deepcopy(
            self.classification_table.loc[:, ~self.classification_table.columns.isin( ["spectrum_iloc", "feature_id"])]
        )
        return output_table
    def get_tsne_coordinates_table(self) -> pd.DataFrame:
        ''' Getter for t-sne coordinates table that attaches the highlight table if available or adds a default. '''        
        output_table = copy.deepcopy(self.tsne_coordinates_table)
        output_table['highlight_bool'] = copy.deepcopy(
            self.highlight_table["highlight_bool"]
        )
        return output_table
    def get_metadata_table(self) -> pd.DataFrame:
        """ Returns a copy of the metadata table. """
        output_table = copy.deepcopy(self.metadata_table)
        return output_table
    def get_spectrum_iloc_list(self) -> List[int]:
        """ Return list of all spectrum_iloc """
        spectrum_ilocs = [elem.spectrum_iloc for elem in self.spectra_specxplore]
        return spectrum_ilocs
    def scale_coordinate_system(self, scaler : float) -> None:
        """ Applies scaling to coordinate system in tsne_coordinates_table in place """
        assert not np.isclose([scaler], [0], rtol=1e-05, atol=1e-08, equal_nan=False)[0], (
            'Scaling with 0 or near 0 not allowed; likely loss of data!'
        )
        self.tsne_coordinates_table["x"] = _scale_array_to_minus1_plus1(
            self.tsne_coordinates_table["x"].to_numpy()
            ) * scaler
        self.tsne_coordinates_table["y"] = _scale_array_to_minus1_plus1(
            self.tsne_coordinates_table["y"].to_numpy()
            ) * scaler
        return None
    
def _scale_array_to_minus1_plus1(array : np.ndarray) -> np.ndarray:
    """ Rescales array to lie between -1 and 1."""
    out = 2.*(array - np.min(array))/np.ptp(array)-1
    return out
def load_specxplore_object_from_pickle(filepath : str) -> SpecxploreSessionData:
    """ Function loads specXplore object from pickle and checks type validity. """
    with open(filepath, 'rb') as file:
        specxplore_object = pickle.load(file) 
    assert isinstance(specxplore_object, SpecxploreSessionData), (
        'Provided data must be a SpecxploreSessionData object!'
    )
    return specxplore_object
