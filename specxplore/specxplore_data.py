from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import namedtuple
import typing
from typing import List, TypedDict, Tuple, Dict, NamedTuple
import copy
from specxplore import specxplore_data_cython

@dataclass
class specxplore_data:
  def __init__(
    self, ms2deepscore_sim, spec2vec_sim, cosine_sim, 
    tsne_df, class_table, is_standard, spectra, mz, specxplore_id
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
    self.class_table = class_table
    self.is_standard = is_standard
    spectra_converted = [
        Spectrum(spec.peaks.mz, spec.get("precursor_mz"), idx, spec.peaks.intensities) 
        for idx, spec in enumerate(spectra)]
    self.spectra = spectra_converted
    self.mz = mz # precursor mz values for each spectrum
    self.specxplore_id = specxplore_id
    # CONSTRUCT SOURCE, TARGET AND VALUE ND ARRAYS
    sources, targets, values = specxplore_data_cython.construct_long_format_sim_arrays(ms2deepscore_sim)
    ordered_index = np.argsort(-values)
    sources = sources[ordered_index]
    targets = targets[ordered_index]
    values = values[ordered_index]
    self.sources = sources
    self.targets = targets
    self.values = values

    self.metadata = pd.concat([tsne_df, class_table], axis=1)



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
    intensities : np.ndarray = None
    # TDOD fix tuple to list of list
    mass_to_charge_ratio_aggregate_list : field(default_factory=tuple) = ()
    intensity_aggregate_list : field(default_factory=tuple) = ()
    is_binned_spectrum : bool = False
    is_neutral_loss : bool = False
    
    def __post_init__(self):
        """ Assert that data provided to constructor is valid. """
        if self.intensities is None:
            self.intensities = np.repeat(np.nan, self.mass_to_charge_ratios.size)
        assert self.intensities.shape == self.mass_to_charge_ratios.shape, (
            "Intensities (array) and mass to charge ratios (array) must be equal shape.")
        if (self.intensity_aggregate_list) and (self.mass_to_charge_ratio_aggregate_list):
            self.is_binned_spectrum = True
            assert len(self.mass_to_charge_ratio_aggregate_list) == len(self.intensity_aggregate_list), (
                "Bin data lists of lists must be of equal length.")
            for x,y in zip(self.intensity_aggregate_list, self.mass_to_charge_ratio_aggregate_list):
                assert len(x) == len(y), ("Sub-lists of aggregate lists must be of equal length, i.e. for each"
                    " mass-to-charge-ratio there must be an intensity value at equal List[sublist] position.")


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