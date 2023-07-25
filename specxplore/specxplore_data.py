from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from collections import namedtuple
import typing
from typing import List, TypedDict, Tuple, Dict, NamedTuple, Union
import copy
from specxplore import specxplore_data_cython
import specxplore.importing
from specxplore import other_utils
from specxplore.clustnet import SELECTED_NODES_STYLE, GENERAL_STYLE, SELECTION_STYLE
import os
import json 
import pickle
import matchms

from kmedoids import KMedoids
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr  
from sklearn.manifold import TSNE
import plotly.graph_objects as go

@dataclass
class KmedoidGridEntry():
    """ 
    Container Class for K medoid clustering results.

    Parameters:
        k: the number of clusters set.
        cluster_assignments: List with cluster assignment for each observation.
        silhouette_score: float with clustering silhouette score.
    """
    k : int
    cluster_assignments : List[int]
    silhouette_score : float
    random_seed_used : int
    def __str__(self) -> str:
        """ Custom Print Method for kmedoid grid entry producing an easy readable string output. """
        custom_print = (
            f"k = {self.k}, silhoutte_score = {self.silhouette_score}, \n"
            f"cluster_assignment = {', '.join(self.cluster_assignments[0:7])}...")
        return custom_print
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
    spectrum_iloc : np.int64
    intensities : np.ndarray
    feature_id : np.string_

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

@dataclass
class specxplore_session_data:
    ''' specxplore_session_data is a constructor class that allow creating all variables for running specxplore 
    dashboards. It comprises of a initiator making use of a matchms spectrum list and a path to a model folder
    to construct pairwise similarity matrices, define spectrum_iloc and feature_id mapping, and constructs
    the list of specXplore spectra used within the dashboard visualizations. Any spectra data processing is assumed
    to have been done before initating the specxplore_session_data object.
    
    A number of essential variables for specXplore are left as None after initition and have to be constructed
    using additional information. The sequence of calls in general is as follows:

    1) Run the tsne-grid and select a tsne coordinate system using self.attach_tsne_grid() and 
       self.select_tsne_coordinates()
    2) Run the kmedoid-grid and select a single or range of k classifications to add to the class table via
       self.attach_kmedoid_grid() and self.select_kmedoid_cluster_assignments()
    3) Attach any metadata from ms2query or elsewhere via self.attach_addon_data_to_metadata()
    4) Attach any additional class table variables via self.attach_addon_data_to_class_table()
    5) Initialize dashboard visual and network variables once all is information included into the session data via
       self.initialize_specxplore_session()
    
    '''
    def __init__(self,spectra_list_matchms: List[matchms.Spectrum], models_and_library_folder_path : str):
        ''' Constructs Basic Scaffold for specXplore session without coordinate system or any metadata. '''
        
        # Making sure that the spectra provided are valid and contain all required information:
        for spectrum in spectra_list_matchms:
            assert spectrum is not None, (
                "None object detected in spectrum list. All spectra must be valid matchms.Spectrum instances.")
            assert spectrum.get("feature_id") is not None, (
                "All spectra must have valid feature id entries.")
            assert spectrum.get("precursor_mz") is not None, (
                "All spectra must have valid precursor_mz value.")
        feature_ids = [str(spec.get("feature_id")) for spec in spectra_list_matchms]
        assert len(feature_ids) == len(set(feature_ids)), ("All feature_ids must be unique.")
        
        # Convert spectra for specXplore visualization modules
        self.spectra = convert_matchms_spectra_to_specxplore_spectra(spectra_list_matchms)
        self.init_table = construct_init_table(self.spectra)

        # Construct pairwise similarity matrices from matchms spectra
        self.scores_spec2vec = specxplore.importing.compute_similarities_s2v(
            spectra_list_matchms, models_and_library_folder_path)
        self.scores_modified_cosine = specxplore.importing.compute_similarities_cosine(
            spectra_list_matchms, cosine_type="ModifiedCosine")
        self.scores_ms2deepscore = specxplore.importing.compute_similarities_ms2ds(
            spectra_list_matchms, models_and_library_folder_path)

        # Initialize data tables to none
        self.metadata_table = copy.deepcopy(self.init_table)
        self.tsne_coordinates_table = None
        self.class_table = None # includes feature_id and spectrum_iloc inside specXplore, but getter only returns classification table
        self.highlight_table = None

        self.class_dict = None 
        self.available_classes = None
        self.selected_class_data = None
        self.initial_style = None
        self.initial_node_elements = None
    
    def initialize_specxplore_session(self) -> None:
        ''' Wrapper for cosmetic and quantitative network variable initialization based on input data. '''
        self.initialize_specxplore_dashboard_variables()
        self.construct_derived_network_variables()
        self.initial_node_elements = other_utils.initialize_cytoscape_graph_elements(
            self.tsne_coordinates_table, self.selected_class_data, self.highlight_table['highlight_bool'].to_list())
        return None


    def initialize_specxplore_dashboard_variables(self):
        ''' Construct variables derived from input that are used inside the dashboard. 
        These will be internal, private style variables, left accessible to the user however. '''
        class_table = self.get_class_table()
        self.class_dict = {elem : list(class_table[elem]) for elem in class_table.columns} 
        self.available_classes = list(self.class_dict.keys())
        self.selected_class_data = self.class_dict[self.available_classes[0]] # initialize default
        self.initial_style = SELECTED_NODES_STYLE + GENERAL_STYLE + SELECTION_STYLE
    

    def construct_derived_network_variables(self) -> None:
        ''' Construct the edge lists and include init styles for specxplor dashboard '''
        sources, targets, values = specxplore_data_cython.construct_long_format_sim_arrays(self.scores_ms2deepscore)  
        ordered_index = np.argsort(-values)
        sources = sources[ordered_index]
        targets = targets[ordered_index]
        values = values[ordered_index]
        self.sources = sources
        self.targets = targets
        self.values = values
        return None
    
    def attach_addon_data_to_metadata(self, addon_data : pd.DataFrame) -> None:
        self.metadata_table = attach_columns_via_feature_id(self.metadata_table, addon_data)
        return None
    

    def attach_addon_data_to_class_table(self, addon_data : pd.DataFrame) -> None:
        if self.class_table is None:
            self.class_table = copy.deepcopy(self.init_table)
        self.class_table = remove_white_space_from_df(attach_columns_via_feature_id(self.class_table, addon_data))
        return None


    def attach_run_tsne_grid(self, perplexity_values : List[int], random_states : Union[List, None] = None) -> None:
        # Generate and attach t-SNE grid
        distance_matrix = convert_similarity_to_distance(self.scores_ms2deepscore)
        self.tsne_grid = run_tsne_grid(distance_matrix, perplexity_values, random_states)
        print_tsne_grid(self.tsne_grid)
        return None


    def attach_kmedoid_grid(self, k_values : List[int], random_states : Union[List, None] = None) -> None:
        distance_matrix = convert_similarity_to_distance(self.scores_ms2deepscore)
        self.kmedoid_grid = run_kmedoid_grid(distance_matrix, k_values, random_states) 
        print_kmedoid_grid(self.kmedoid_grid)
        return None


    def select_tsne_coordinates(self, scalar_iloc : int) -> None:
        """ Select a tsne coordinate system from kmedoid grid via iloc in grid list. """
        assert self.tsne_grid is not None, (
            'No tsne grid detected. Run attach_tsne_grid to be able to'
            ' be able select a kmedoid grid entry.')
        assert isinstance(scalar_iloc, int), 'scalar_iloc must be single type int variable'
        assert scalar_iloc in [i for i in range(0, len(self.tsne_grid))]
        # construct tsne table
        if self.tsne_coordinates_table is None:
            self.tsne_coordinates_table = copy.deepcopy(self.init_table)
        # Extract relevant tsne_grid entry & attach coordinates
        tsne_entry = self.tsne_grid[scalar_iloc]
        # only t-sne coordinates are overwritten
        self.tsne_coordinates_table ['x'] = tsne_entry.x_coordinates
        self.tsne_coordinates_table ['y'] = tsne_entry.y_coordinates
        return None


    def select_kmedoid_cluster_assignments(self, iloc_list : List[int]):
        """ Select one or more kmedoid k levels for class table via ilocs in grid list. """
        assert isinstance(iloc_list, list), 'iloc list must be type list. If only one value, use [value]'
        assert self.kmedoid_grid is not None, (
            'No kmedoid grid detected. Run attach_kmedoid_grid to be able to'
            ' be able select a kmedoid grid entry.')
        for iloc in iloc_list:
            assert isinstance(iloc, int), 'iloc must be single type int variable'
            assert iloc in [i for i in range(0, len(self.kmedoid_grid))]
        # construct kmedoid class table / attach to class_tablw
        if self.class_table is None:
            self.class_table = copy.deepcopy(self.init_table)
        
        selected_subgrid = [self.kmedoid_grid[iloc] for iloc in iloc_list]
        kmedoid_table = pd.DataFrame({"K = " + str(elem.k) : elem.cluster_assignments for elem in selected_subgrid})
        
        kmedoid_table = kmedoid_table.loc[:, ~kmedoid_table.columns.isin(self.class_table.columns.to_list())]
        self.class_table = pd.concat([self.class_table, kmedoid_table], axis=1, join='inner')
        return None
    

    def reset_class_table(self):
        """ Resets specXplore class_table entry to None. """
        self.class_table = None
        return None
    

    def reset_metadata_table(self):
        """ Resets specXplore class_table entry to None. """
        self.metadata_table = None
        return None
    

    def get_tsne_coordinates_table(self):
        ''' Getter for t-sne coordinates table that attaches the highlight table if available or adds a default. '''
        # return tsne table, add in the highlight table if not none, if none, add a highlight all false column
        assert self.tsne_coordinates_table is not None, 'tsne_coordinates_table does not exist and cannot be returned.'
        output_table = copy.deepcopy(self.tsne_coordinates_table)
        if self.highlight_table is None:
            output_table['highlight_bool'] = False
        else:
            output_table['highlight_bool'] = copy.deepcopy(self.highlight_table["highlight_bool"])
        return output_table
    

    def get_class_table(self):
        assert self.class_table is not None, 'class_table does not exist and cannot be returned.'
        output_table = copy.deepcopy(self.class_table.loc[:, ~self.class_table.columns.isin( ["spectrum_iloc", "feature_id"])])
        return output_table
    

    def get_metadata_table(self):
        assert self.metadata_table is not None, 'class_table does not exist and cannot be returned.'
        output_table = copy.deepcopy(self.metadata_table)
        return output_table
    
    
    def construct_highlight_table(self, feature_ids : List[str]) -> None:
        ''' Construct the table of features considered knowns or standards for visual highlighting in specXplore overview. 
        
        Input:
            feature_id: list of str entries specifying the feature_ids worth highlighting in specXplore. Usually 
                spike-in standards or high confidence library matches.
        Attaches:
            highlight table with features to be highlighted.

        Developer Notes:
            This function can be used with init tables, but also with the t-SNE table directly sicne the relevant entries
            are available.
        Requires a init table and feature_ids designated for highlighting.
        '''
        feature_set = set(feature_ids)
        highlight_table = copy.deepcopy(self.init_table)
        highlight_table['highlight_bool'] = [elem in feature_set for elem in self.init_table["feature_id"]]
        self.highlight_table = highlight_table
        return None
    

    def get_spectrum_iloc_list(self) -> List[int]:
        """ Return list of all spectrum_iloc """
        return self.init_table['spectrum_iloc'].to_list()

    def check_and_save_to_file(self, filepath : str) -> None:
        """ Saves specxplore data object using pickle provided all data elements available."""
        assert self.class_table is not None, 'class_table not found. incomplete specxplore object cannot be saved or loaded'
        assert self.highlight_table is not None, 'highlight_table not found. incomplete specxplore object cannot be saved or loaded'
        assert self.metadata_table is not None, 'metadata_table not found. incomplete specxplore object cannot be saved or loaded'
        assert self.tsne_coordinates_table is not None, 'tsne_coordinates_table not found. incomplete specxplore object cannot be saved or loaded'
        assert self.values is not None, 'values not found. incomplete specxplore object cannot be saved or loaded'
        assert self.targets is not None, 'targets not found. incomplete specxplore object cannot be saved or loaded'
        assert self.sources is not None, 'sources not found. incomplete specxplore object cannot be saved or loaded'
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)
        return None


    def save_selection_to_file(self, filepath : str, selection_idx : List[int]) -> None:
        ''' Functions copies current session_data objects and replaces all member variables with subselection
        before saving to file. Overwriting a copy is done to avoid making the user facing constructor more 
        complicated (overloading not possible in python)'''
        
        assert len(selection_idx) >= 2, "specXplore object requires at least 2 spectra to be selected."

        # subset data structures
        scores_ms2deepscore = self.scores_ms2deepscore[selection_idx, :][:, selection_idx].copy()
        scores_spec2vec = self.scores_spec2vec[selection_idx, :][:, selection_idx].copy()
        scores_modified_cosine = self.scores_modified_cosine[selection_idx, :][:, selection_idx].copy()
        
        tsne_coordinates_table = self.tsne_coordinates_table.iloc[selection_idx].copy()
        
        metadata_table = self.metadata_table.iloc[selection_idx].copy()
        class_table = self.class_table.iloc[selection_idx].copy()
        init_table = self.init_table.iloc[selection_idx].copy()
        highlight_table = self.highlight_table.iloc[selection_idx].copy()

        new_spectrum_iloc = [idx for idx in range(0, len(selection_idx))]
        metadata_table['spectrum_iloc'] = new_spectrum_iloc
        metadata_table.reset_index(drop=True, inplace=True)
        class_table['spectrum_iloc'] = new_spectrum_iloc
        class_table.reset_index(drop=True, inplace=True)
        init_table['spectrum_iloc'] = new_spectrum_iloc
        init_table.reset_index(drop=True, inplace=True)
        highlight_table['spectrum_iloc'] = new_spectrum_iloc
        highlight_table.reset_index(drop=True, inplace=True)
        
        spectra = copy.deepcopy(self.spectra) # make a deep copy to detach from actual spectrum list
        spectra = [self.spectra[idx] for idx in selection_idx] # subset spectrum list


        new_specxplore_session = copy.deepcopy(self)
        new_specxplore_session.scores_ms2deepscore = scores_ms2deepscore
        new_specxplore_session.scores_modified_cosine = scores_modified_cosine
        new_specxplore_session.scores_spec2vec = scores_spec2vec
        new_specxplore_session.tsne_coordinates_table = tsne_coordinates_table # beware: tsne coordinates not optimal for sub-selection
        new_specxplore_session.metadata_table = metadata_table
        new_specxplore_session.class_table = class_table
        new_specxplore_session.init_table = init_table
        new_specxplore_session.highlight_table = highlight_table
        new_specxplore_session.spectra = spectra
        new_specxplore_session.initialize_specxplore_session()
        
        with open(filepath, "wb") as file:
            pickle.dump(new_specxplore_session, file)
        return None
    

    def save_pairwise_similarity_matrices_to_file(self, run_name : str, directory_path : str) -> None:
        """ Saves the three similarity matrices to file with a run_name prefix to the specified directory. The output 
        format is a .npy object that can be loaded using numpy.load.
        """
        np.save(os.path.join(directory_path, run_name, "ms2ds.npy"), self.scores_ms2deepscore, allow_pickle=False)
        np.save(os.path.join(directory_path, run_name, "modcos.npy"), self.scores_modified_cosine, allow_pickle=False)
        np.save(os.path.join(directory_path, run_name, "s2v.npy"), self.scores_spec2vec, allow_pickle=False)
        return None
    
    def scale_coordinate_system(self, scaler : float):
        """ Applies scaling to coordinate system in tsne_coordinates_table """
        assert not np.isclose([scaler], [0], rtol=1e-05, atol=1e-08, equal_nan=False)[0], (
            'Scaling with 0 or near 0 not allowed; likely loss of data!')
        self.tsne_coordinates_table["x"] = other_utils.scale_array_to_minus1_plus1(
             self.tsne_coordinates_table["x"].to_numpy()) * scaler
        self.tsne_coordinates_table["y"] = other_utils.scale_array_to_minus1_plus1(
             self.tsne_coordinates_table["y"].to_numpy()) * scaler


def convert_matchms_spectra_to_specxplore_spectra(spectra = List[matchms.Spectrum]) -> List[Spectrum]:
  """ Converts list of matchms.Spectrum objects to list of specxplore_data.Spectrum objects. """
  spectra_converted = [
      Spectrum(
        mass_to_charge_ratios = spec.peaks.mz, 
        precursor_mass_to_charge_ratio = float(spec.get("precursor_mz")), 
        spectrum_iloc = idx, 
        intensities = spec.peaks.intensities, 
        feature_id=spec.get("feature_id")) 
      for idx, spec in enumerate(spectra)]
  return spectra_converted


def construct_init_table(spectra : List[Spectrum]) -> pd.DataFrame:
    ''' Creates initialized table for metadata or classification in specXplore. Table is a pandas.DataFrame with
    string and int columns indicating the feature_id, and spectrum_iloc.
    
    Parameters
        spectra: alist of matchms.spectrum objects. These should contain spectra with unique feature_ids.

    Returns
        init_table: a pandas.DataFrame with two columns: a string column for feature_id, and a int column for 
        spectrum_iloc.
    '''
    spectrum_ilocs = [spec.spectrum_iloc for spec in spectra]
    feature_ids = [spec.feature_id for spec in spectra] 
    assert spectrum_ilocs == [iloc for iloc in range(0, len(spectra))], (
        "spectrum iloc must equal sequence from 0 to number of spectra")
    init_table = pd.DataFrame({"feature_id" : feature_ids, "spectrum_iloc" : spectrum_ilocs})
    init_table["feature_id"] = init_table["feature_id"].astype("string")
    return init_table


def load_specxplore_object_from_pickle(filepath : str) -> specxplore_session_data:
    with open(filepath, 'rb') as file:
        specxplore_object = pickle.load(file) 
    assert isinstance(specxplore_object, specxplore_session_data)
    return specxplore_object


def filter_spectrum_top_k_intensity_fragments(input_spectrum : Spectrum, k : int) -> Spectrum:
    """ Filter unbinned Spectrum object to top-K highest intensity fragments for display in fragmap. """
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
            spectrum_iloc = spectrum.spectrum_iloc, 
            feature_id = spectrum.feature_id, 
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


def run_tsne_grid(distance_matrix : np.ndarray, perplexity_values : List[int], random_states : Union[List, None] = None) -> List[TsneGridEntry]:
    """ Runs t-SNE embedding routine for every provided perplexity value in perplexity_values list.

    Parameters:
        distance_matrix: An np.ndarray containing pairwise distances.
        perplexity_values: A list of perplexity values to try for t-SNE embedding.
        random_states: None or a list of integers specifying the random state to use for each k-medoid run.
    Returns: 
        A list of TsneGridEntry objects containing grid results. 
    """
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
    """ Plots pearson and spearman scores vs perplexity for each entry in list of TsneGridEntry objects. """
    pearson_scores = [x.spearman_score for x in tsne_list]
    spearman_scores = [x.pearson_score for x in tsne_list]
    perplexities = [x.perplexity for x in tsne_list]

    trace_spearman = go.Scatter(x = perplexities, y = spearman_scores, name="spearman_score", mode = "markers")
    trace_pearson = go.Scatter(x = perplexities, y = pearson_scores, name="pearson_score", mode = "markers")
    fig = go.Figure([trace_pearson, trace_spearman])
    fig.update_layout(xaxis_title="Perplexity", yaxis_title="Score")
    fig.show(renderer = "browser")
    return None


def convert_similarity_to_distance(similarity_matrix : np.ndarray) -> np.ndarray:
    """ 
    Converts pairwise similarity matrix to distance matrix with values between 0 and 1. Assumes that the input is a
    similarity matrix with values in range 0 to 1 up to floating point error.

    Developer Note:
        spec2vec scores do not appear to be in this range.

    """
    distance_matrix = 1.- similarity_matrix
    distance_matrix = np.round(distance_matrix, 6) # Round to deal with floating point issues
    distance_matrix = np.clip(distance_matrix, a_min = 0, a_max = 1) # Clip to deal with floating point issues
    return distance_matrix


def run_kmedoid_grid(distance_matrix : np.ndarray, k_values : List[int], random_states : Union[List, None] = None) -> List[KmedoidGridEntry]:
    """ Runs k-medoid clustering for every value in k_values. 
    
    Parameters:
        distance_matrix: An np.ndarray containing pairwise distances.
        k_values: A list of k values to try in k-medoid clustering.
        random_states: None or a list of integers specifying the random state to use for each k-medoid run.
    Returns: 
        A list of KmedoidGridEntry objects containing grid results.
    """
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
    """ Plots Silhouette Score vs k for each entry in list of KmedoidGridEntry objects. """
    scores = [x.silhouette_score for x in kmedoid_list]
    ks = [x.k for x in kmedoid_list]
    fig = px.scatter(x = ks, y = scores)
    fig.update_layout(xaxis_title="K (Number of Clusters)", yaxis_title="Silhouette Score")
    fig.show(renderer = "browser")
    return None


def print_kmedoid_grid(grid : List[KmedoidGridEntry]) -> None:
    print("iloc Number-of-Clusters Silhouette-Score")
    for iloc, elem in enumerate(grid):
        print(iloc, elem.k, round(elem.silhouette_score, 3))
    return None


def print_tsne_grid(grid : List[TsneGridEntry]) -> None:   
    print('iloc Perplexity Pearson-score Spearman-score')
    for iloc, elem in enumerate(grid):
        print(iloc, elem.perplexity, round(elem.pearson_score, 3), round(elem.spearman_score, 3))


def attach_columns_via_feature_id(init_table : pd.DataFrame, addon_data : pd.DataFrame,) -> pd.DataFrame:
    """ Attaches addon_data to data frame via join on 'feature_id'. 
    
    The data frame can be a class table or metadata table and is assumed to be derived from the construct_init_table() 
    and contain feature_id and spectrum_iloc columns.
    
    Input
        init_table: pandas.DataFrame object with at least a feature_id column (type string)
        addon_data: pandas.DataFrame object with a feature_id column and additional columns to be merged into metadata. 
            Columns can be of any type.
    Output: 
        extended_init_table: pandas.DataFrame with feature_id column and additional columns from addon_data. Any NA values
            produced are replaced with strings that read: "not available". Any entries are converted to string.

    """
    assert "feature_id" in init_table.columns, "feature_id column must be available in metadata"
    assert "feature_id" in addon_data.columns, "feature_id column must be available in addon_data"
    # there is no decent means of checking whether a pandas data frame is of proper type string apparently
    assert init_table["feature_id"].dtype == addon_data["feature_id"].dtype == 'string', "feature_id column must be of the same type."

    
    extended_init_table = copy.deepcopy(init_table)
    extended_init_table = extended_init_table.merge(
        addon_data.loc[:, ~addon_data.columns.isin(['spectrum_iloc'])],
        left_on = "feature_id", right_on = "feature_id", how = "left")
    extended_init_table.reset_index(inplace=True, drop=True)
    extended_init_table = extended_init_table.astype('string')
    extended_init_table = extended_init_table.replace(to_replace=np.nan, value = "not available")
    return extended_init_table


def remove_white_space_from_df(input_df : pd.DataFrame) -> pd.DataFrame:
    ''' Removes whitespace from all entries in input_df. 
    
    White space removal is essential for accurate chemical classification parsing in node highlighting of specXplore.
    '''
    output_df = copy.deepcopy(input_df)
    output_df = input_df.replace(to_replace=" ", value = "_", regex=True)
    return output_df




