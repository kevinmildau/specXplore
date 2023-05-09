from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import namedtuple
import typing
from typing import List, TypedDict, Tuple, Dict, NamedTuple, Union
import copy
from specxplore import specxplore_data_cython
from specxplore import other_utils
from specxplore.clustnet import SELECTED_NODES_STYLE, GENERAL_STYLE, SELECTION_STYLE
import os
import json 
import pickle
@dataclass
class specxplore_data:
    """ specxplore data container that supplies all the data needed by the specxplore dashboard.
    
    
    Developer Notes:
    loading and saving specxplore objects is done using pickle. This is because many of the structures in specxplore
    are not json serializable, and would require elaborate conversions and parsing to work:
        pandas df
        numpy arrays
        Spectrum object (including numpy arrays)
    """
    def __init__(
        self, ms2deepscore_sim, spec2vec_sim, cosine_sim, 
        tsne_df, class_table, is_standard, spectra, mz, specxplore_id, metadata
        ):
        self.ms2deepscore_sim = ms2deepscore_sim
        self.spec2vec_sim = spec2vec_sim
        self.cosine_sim = cosine_sim
        tsne_df["is_standard"] = is_standard
        tsne_df["id"] = specxplore_id

        self.tsne_df = tsne_df
        class_table.columns = class_table.columns.astype(str)
        class_table.astype(str)
        class_table = class_table.replace(" ","_", regex = True)
        class_table["is_standard"] = pd.Series(is_standard, dtype = str) # tmp modification
        self.class_table = class_table
        
        self.class_dict = {elem : list(class_table[elem]) for elem in class_table.columns} 
        self.available_classes = list(self.class_dict.keys())
        self.selected_class_data = self.class_dict[self.available_classes[0]] # initialize default
        self.is_standard = is_standard


        self.spectra = spectra
        self.mz = mz # precursor mz values for each spectrum
        self.specxplore_ids = specxplore_id
        # CONSTRUCT SOURCE, TARGET AND VALUE ND ARRAYS
        sources, targets, values = specxplore_data_cython.construct_long_format_sim_arrays(ms2deepscore_sim)
        ordered_index = np.argsort(-values)
        sources = sources[ordered_index]
        targets = targets[ordered_index]
        values = values[ordered_index]
        self.sources = sources
        self.targets = targets
        self.values = values

        self.metadata = metadata
        self.initial_node_elements = other_utils.initialize_cytoscape_graph_elements(
            self.tsne_df, self.selected_class_data, self.is_standard)
        self.initial_style = SELECTED_NODES_STYLE + GENERAL_STYLE + SELECTION_STYLE

    def scale_coordinate_system(self, scaler : float):
        """ Applies scaling to coordinate system """
        self.tsne_df["x"] = other_utils.scale_array_to_minus1_plus1(self.tsne_df["x"].to_numpy()) * scaler
        self.tsne_df["y"] = other_utils.scale_array_to_minus1_plus1(self.tsne_df["y"].to_numpy()) * scaler


    def save_to_file(self, filepath : str) -> None:
        """ Saves specxplore data object as json string. """
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        return None

    def save_selection_to_file(self, filepath : str, selection_idx : List[int]) -> None:
        # extract selections from matrices and numpy arrays. 
        # reindex selections using new specXplore ids
        # keep a data_origin id somewhere (from original feature table)
        assert len(selection_idx) >= 2, "specXplore object requires at least 2 spectra to be selected."
        assert set(selection_idx).issubset(set(self.specxplore_ids)), "selection set must be contained within current specxplore set"
        # subset data structures
        ms2deepscore_sim = self.ms2deepscore_sim[selection_idx, :][:, selection_idx].copy()
        spec2vec_sim = self.spec2vec_sim[selection_idx, :][:, selection_idx].copy()
        cosine_sim = self.cosine_sim[selection_idx, :][:, selection_idx].copy()
        tsne_df = self.tsne_df.iloc[selection_idx].copy()
        tsne_df.reset_index(drop=True, inplace=True)
        class_table = self.class_table.iloc[selection_idx].copy()
        class_table.reset_index(drop=True, inplace=True)
        metadata = self.metadata.iloc[selection_idx].copy()
        metadata.reset_index(drop=True, inplace=True)
        is_standard = [self.is_standard[idx] for idx in selection_idx]
        


        mz = [self.mz[idx] for idx in selection_idx]

        # update t-sne df ids and construct new specxplore id list
        n_spectra = len(selection_idx)
        specxplore_ids = [idx for idx in range(0, len(selection_idx))]

        feature_id_mapping_old_vs_new = {selection_idx[idx] : idx for idx in range(0, n_spectra)}
        old_id_array = tsne_df["id"].to_numpy().copy()

        new_ids = np.array([feature_id_mapping_old_vs_new[x] for x in old_id_array])
        tsne_df = tsne_df.assign(id=new_ids)

        # update spectrum identifier to appropriate specxplore_id
        # needs new ids
        # needs old ids
        # or id mapping
        spectra = copy.deepcopy(self.spectra) # make a deep copy to detach from actual spectrum list
        spectra = [self.spectra[idx] for idx in selection_idx] # subset spectrum list
        for spectrum in spectra:
            spectrum.identifier = feature_id_mapping_old_vs_new[spectrum.identifier] # gets the id corresponding to old id

        specxplore_object = specxplore_data(
            ms2deepscore_sim, spec2vec_sim, cosine_sim, tsne_df, class_table, is_standard, spectra, mz, specxplore_ids, metadata)
        
        with open(filepath, "wb") as file:
            pickle.dump(specxplore_object, file)
        return None


def load_specxplore_object_from_pickle(filepath : str) -> specxplore_data:
    with open(filepath, 'rb') as file:
        specxplore_object = pickle.load(file) 
    assert isinstance(specxplore_object, specxplore_data)
    return specxplore_object


@dataclass
class Spectrum:
    """ Spectrum data class for storing basic spectrum information and neutral loss spectra. 

    Parameters:
    :param mass_to_charge_ratio: np.ndarray of shape(1,n) where n is the number of mass to charge ratios.
    :param precursor_mass_to_charge_ratio: np.double with mass to charge ratio of precursor.
    :param identifier: np.int64 is the spectrum's identifier number.
    :param intensities: np.ndarray of shape(1,n) where n is the number of intensities.
    :param mass_to_charge_ratio_aggregate_list: List[List] containing original mass to charge ratios merged 
        together during binning.
    :param intensity_aggregate_list: List[List] containing original intensity values merged together during binning.
    :param binned_spectrum: Bool, autodetermined from presence of aggregate lists to specify that the spectrum has been 
        binned.
    Raises:
        ValueError: if shapes of intensities and mass_to_charge_ratio arrays differ.
        ValueError: if length mismatches in List[List] structures for aggregate lists if provided.
    
    Developer Notes: 
    Spectrum identifier should correspond to the iloc of the spectrum in the orginal spectrum list used in specxplore.
    There are no checks in place within the Spectrum object to make sure this is the case.
    Intensities are not necessary as an input. This is to accomodate neutral loss mock spectra objects. If no 
    intensities are provided, intensity values are set to np.nan assuming neutral loss spectra were provided.
    Aggregate lists are an optional input an by-product of binning. If no two mz values were put into the same mass-
    to-charge-ratio bin, then the aggregate lists contains only lists of len 1.
    """
    mass_to_charge_ratios : np.ndarray #np.ndarray[int, np.double] # for python 3.9 and up
    precursor_mass_to_charge_ratio : np.double
    identifier : np.int64
    intensities : np.ndarray

    # TDOD fix tuple to list of list
    mass_to_charge_ratio_aggregate_list : field(default_factory=tuple) = ()
    intensity_aggregate_list : field(default_factory=tuple) = ()
    is_binned_spectrum : bool = False
    is_neutral_loss : bool = False
    
    def __post_init__(self):
        """ Assert that data provided to constructor is valid. """
        assert self.intensities.shape == self.mass_to_charge_ratios.shape, (
            "Intensities (array) and mass to charge ratios (array) must be equal shape.")
        if (self.intensity_aggregate_list) and (self.mass_to_charge_ratio_aggregate_list):
            self.is_binned_spectrum = True
            assert len(self.mass_to_charge_ratio_aggregate_list) == len(self.intensity_aggregate_list), (
                "Bin data lists of lists must be of equal length.")
            for x,y in zip(self.intensity_aggregate_list, self.mass_to_charge_ratio_aggregate_list):
                assert len(x) == len(y), ("Sub-lists of aggregate lists must be of equal length, i.e. for each"
                    " mass-to-charge-ratio there must be an intensity value at equal List[sublist] position.")

def filter_spectrum_top_k_intensity_fragments(input_spectrum : Spectrum, k : int) -> Spectrum:
    """ Filter unbinned Spectrum object to top-K highest intensity fragments """
    assert k >= 1, 'k must be larger or equal to one.'
    assert input_spectrum.is_binned_spectrum == False, "filter_spectrum_top_k_intensity_fragments() requires unbinned spectrum."
    spectrum = copy.deepcopy(input_spectrum)
    if spectrum.intensities.size > k:
        index_of_k_largest_intensities = np.argpartition(spectrum.intensities, -k)[-k:]
        mass_to_charge_ratios = spectrum.mass_to_charge_ratios[index_of_k_largest_intensities]
        intensities = spectrum.intensities[index_of_k_largest_intensities]
        spectrum = Spectrum(
            mass_to_charge_ratios = mass_to_charge_ratios, 
            precursor_mass_to_charge_ratio = spectrum.precursor_mass_to_charge_ratio,
            identifier = spectrum.identifier, 
            intensities = intensities)
    return(spectrum)
        

@dataclass(frozen=True)
class SpectraDF:
    """ 
    Dataclass container for long format data frame containing multiple spectra.

    Parameters:
        data: A pandas.DataFrame with columns ('spectrum_identifier', 'mass_to_charge_ratio', 'intensity', 
        'mass_to_charge_ratio_aggregate_list', 'intensity_aggregate_list', 'is_neutral_loss', 'is_binned_spectrum') 
        of types (np.int64, np.double, np.double, object, object, bool, bool). For both aggregate_list columns the 
        expected input is a List[List[np.double]].
    Methods:
        get_data(): Returnsa copy of data frame object stored in SpectraDF instance.
        get_column_as_np(): Returns a copy of a specific column from SpectraDF as numpy array.

    Developer Note: 
        Requires assert_column_set and assert_column_types functions.
        The data dataframe elements are still mutable, frozen only prevent overwriting the object as a whole. Accidental
        modification can be prevented by using the get_data() method and avoiding my_SpectraDF._data accessing.
    """
    _data: pd.DataFrame
    _expected_columns : Tuple = field(
        default=('spectrum_identifier', 'mass_to_charge_ratio', 'intensity', 'mass_to_charge_ratio_aggregate_list', 
            'intensity_aggregate_list', 'is_neutral_loss', 'is_binned_spectrum'), 
        compare = False, hash = False, repr=False)
    _expected_column_types : Tuple = field(
        default=(np.int64, np.double, np.double, object, object, bool, bool), 
        compare = False, hash = False, repr=False )    
    def __post_init__(self):
        """ Assert that data provided to constructor is valid. """
        assert isinstance(self._data, pd.DataFrame), "Data must be a pandas.DataFrame"
        expected_column_types = dict(zip(self._expected_columns, self._expected_column_types))
        assert_column_set(self._data.columns.to_list(), self._expected_columns)
        assert_column_types(self._data.dtypes.to_dict(), expected_column_types)
    def get_data(self):
        """ Return a copy of the data frame object stored in SpectraDF instance. """
        return copy.deepcopy(self._data)
    def get_column_as_np(self, column_name):
        """ Return a copy of a specific column from SpectraDF as numpy array. """
        assert column_name in self._expected_columns, f"Column {column_name} not a member of SpectraDF data frame."
        array = self._data[column_name].to_numpy(copy=True)
        return array

def assert_column_types(type_dict_provided , type_dict_expected ) -> None:
    """ 
    Assert types for keys in type_dict match those for key in expected.

    Parameters:
        type_dict_provided: Dict with key-value pairs containing derived column name (str) and column type (type).
        type_dict_expected: Dict with key-value pairs containing expected column name (str) and column type (type).
    Returns:
        None
    Raises:
        ValueError: if types do not match in provided dictionaries.
    """
    for key in type_dict_provided:
        assert type_dict_provided[key] == type_dict_expected[key], (
            f"Provided dtype for column {key} is {type_dict_provided[key]},"
            f" but requires {type_dict_expected[key]}")
    return None

def assert_column_set(columns_provided : List[str], columns_expected : List[str]) -> None:
    """
    Check if provided columns match with expected columns.

    Parameters:
        columns_provided: List[str] of column names (columns derived from pd.DataFrame).
        columns_expected: List[str] of column names (columns expected for pd.DataFrame).
    Returns:
        None.
    Raises:
        ValueError: if column sets provided don't match.
    """
    set_provided = set(columns_provided)
    set_expected = set(columns_expected)
    assert set_provided == set_expected, ("Initialization error, provided columns do not match expected set.")
    return None

@dataclass(frozen=True)
class EdgeList:
    """ 
    Immutable container for edge list data comprised of 3 1D numpy arrays: source, targets and edges.

    Parameters:
        sources: 1D np.ndarray with dtype int64 containing integer identifiers for target nodes.
        targets: 1D np.ndarray with dtype int64 containing integer identifiers for target nodes.
        values: 1D np.ndarray with dtype np.double containing edge weight.

    Developer Notes:
    This dataclass itself is frozen meaning that member variables cannot be overwritten. In addition, the use of
    write = False flags for the numpy arrays prevents overwriting of any numpy array elements.
    """
    sources: np.ndarray
    targets: np.ndarray
    values: np.ndarray
    def __post_init__(self):
        """ Check input validity and make numpy arrays immutable. """
        assert self.sources.size == self.targets.size == self.values.size, "Arrays must be equal size."
        size = self.sources.size
        expected_shape_tuple = tuple([size])
        assert (self.sources.shape == expected_shape_tuple 
            and self.targets.shape == expected_shape_tuple 
            and self.values.shape == expected_shape_tuple), (
            f"All input arrays must shape 1 dimensional with shape {expected_shape_tuple}")
        assert self.sources.dtype == np.int64, "Targets array must have dtype.int64"
        assert self.targets.dtype == np.int64, "Sources array must have dtype.int64"
        assert self.values.dtype == np.double, "Values array must have dtype.double"
        # Make numpy arrays immutable
        self.sources.setflags(write=False)
        self.targets.setflags(write=False)
        self.values.setflags(write=False)