from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import namedtuple
import typing
from typing import List, TypedDict, Tuple, Dict, NamedTuple
class specxplore_data:
  def __init__(
    self, ms2deepscore_sim, spec2vec_sim, cosine_sim, 
    tsne_df, class_table, clust_table, is_standard, spectra, mz, specxplore_id
    ):
    self.ms2deepscore_sim = ms2deepscore_sim
    self.spec2vec_sim = spec2vec_sim
    self.cosine_sim = cosine_sim
    self.tsne_df = tsne_df
    #tmp_class_table = class_table.merge(
    #  clust_table, 
    #  how = 'inner', 
    #  on='specxplore_id').drop(["specxplore_id"], axis = 1)
    #tmp_class_table.replace(np.nan, 'Unknown')
    #self.class_table = tmp_class_table
    self.class_table = class_table
    self.is_standard = is_standard
    self.spectra = spectra
    self.mz = mz # precursor mz values for each spectrum
    self.specxplore_id = specxplore_id

@dataclass
class Spectrum:
    """ Spectrum data class for storing basic spectrum information and neutral loss spectra. 

    :param mass_to_charge_ratio: np.ndarray of shape(1,n) where n is the number of mass to charge ratios.
    :param precursor_mass_to_charge_ratio: np.double with mass to charge ratio of precursor.
    :param identifier: np.int64 is the spectrum's identifier number.
    :param intensities: np.ndarray of shape(1,n) where n is the number of intensities.
    :param mass_to_charge_ratio_aggregate_list: List[List] containing original mass to charge ratios merged 
        together during binning.
    :param intensity_aggregate_list: List[List] containing original intensity values merged together during binning.
    :param binned_spectrum: Bool, autodetermined from presence of aggregate lists to specify that the spectrum has been 
        binned.
    :raises: Error if size shapes of intensities and mass_to_charge_ratio arrays differ.
    
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
    mass_to_charge_ratio_aggregate_list : field(default_factory=tuple) = ()
    intensity_aggregate_list : field(default_factory=tuple) = ()
    is_binned_spectrum : bool = False
    is_neutral_loss : bool = False
    def __post_init__(self):
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
class MultiSpectrumDataFrameContainer:
    """ 
    Dataclass container for long format data frame containing multiple spectra.
    :param data: A dataframe with columns ("identifier", "mass-to-charge-ratio", "intensity") of types 
        (np.int64, np.double, np.double).
    Note: 
        Requires check_columns and check_column_types functions.
        The data dataframe elements are still mutable, frozen only prevent overwriting the object as a whole.
    """
    data: pd.DataFrame
    expected_columns : Tuple = field(
        default=("identifier", "mass-to-charge-ratio", "intensity"), 
        compare = False, hash = False, repr=False)
    expected_column_types : Tuple = field(
        default=(np.int64, np.double, np.double), 
        compare = False, hash = False, repr=False )    
    def __post_init__(self):
        # Check whether all provided information is in line with expected values
        # Note that data frame elements will still be mutable.
        expected_column_types = dict(zip(self.expected_columns, self.expected_column_types))
        check_columns(self.data.columns.to_list(), self.expected_columns)
        check_column_types(self.data.dtypes.to_dict(), expected_column_types)
    def get_data(self):
        return self.data
    def get_column_as_np(self, column_name):
        assert column_name in self.expected_columns, f"Column {column_name} not a member of MultiSpectrumDataContainer."
        array = self.data[column_name].to_numpy()
        return array

def check_column_types(type_dict_provided , type_dict_expected ) -> None:
    print("Checkpoint")
    for key in type_dict_provided:
        assert type_dict_provided[key] == type_dict_expected[key], (
            f"Provided dtype for column {key} is {type_dict_provided[key]},"
            f" but requires {type_dict_expected[key]}")
    return None

def check_columns(columns_provided : List[str], columns_expected : List[str]) -> None:
    """
    Check if provided columns match with expected columns.

    :param columns_provided: List[str] of column names (expected derived from pd.DataFrame).
    :param columns_expected: List[str] of column names (expected columns for pd.DataFrame)
    
    :raises: ValueError if column sets provided don't match.
    """
    set_provided = set(columns_provided)
    set_expected = set(columns_expected)
    assert set_provided == set_expected, ("Initialization error, provided columns do not match expected set.")
    return None