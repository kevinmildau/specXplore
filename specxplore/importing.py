from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from typing import List, TypedDict, Tuple, Dict, Union
import copy
from specxplore.session_data import SpecxploreSessionData
from specxplore.spectrum import Spectrum
from specxplore import importing_cython
from specxplore import utils
import os
import json 
import pickle
import matchms
from kmedoids import KMedoids
from sklearn.metrics import silhouette_score
from sklearn.manifold import TSNE
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr  
import plotly.graph_objects as go
import plotly.express
import matchms.utils 
import os
import gensim
from spec2vec import Spec2Vec
import matchms
from matchms import calculate_scores
from matchms.similarity import CosineGreedy, ModifiedCosine, CosineHungarian
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import load_model
from ms2query import SettingsRunMS2Query
from ms2query.run_ms2query import run_ms2query_single_file
from ms2query.ms2library import create_library_object_from_one_dir
import copy
import numpy as np
import pandas as pd
import warnings

@dataclass
class KmedoidGridEntry():
    """ 
    Container Class for K medoid clustering results. Contains a single entry.

    Parameters:
        k: the number of clusters set.
        cluster_assignments: List with cluster assignment for each observation.
        silhouette_score: float with clustering silhouette score.
        random_seed_used : int or float with the random seed used in k-medoid clustering.
    """
    k : int
    cluster_assignments : List[int]
    silhouette_score : float
    random_seed_used : Union[int, float]
    def __str__(self) -> str:
        """ Custom Print Method for kmedoid grid entry producing an easy readable string output. """
        custom_print = (
            f"k = {self.k}, silhoutte_score = {self.silhouette_score}, \n"
            f"cluster_assignment = {', '.join(self.cluster_assignments[0:7])}...")
        return custom_print
@dataclass
class TsneGridEntry():
    """ 
    Container Class for t-SNE embedding optimization results. Contains a single entry.

    Parameters:
        perplexity : int with perplexity value used in t-SNE optimization.
        x_coordinates : List[int] x coordinates produced by t-SNE
        y_coordinates:  List[int] y coordinates produced by t-SNE
        pearson_score : float representing the pearson correlation between pairwise distances in embedding and 
            high dimensional space.
        spearman_score : float representing the spearman correlation between pairwise distances in embedding and 
            high dimensional space.
        random_seed_used : int or float with the random seed used in k-medoid clustering.
    """
    perplexity : int
    x_coordinates : List[int]
    y_coordinates:  List[int]
    pearson_score : float
    spearman_score : float
    random_seed_used : Union[int, float]
    def __str__(self) -> str:
        custom_print = (
            f"Perplexity = {self.perplexity}," 
            f"Pearson Score = {self.pearson_score}, "
            f"Spearman Score = {self.spearman_score}, \n"
            f"x coordinates = {', '.join(self.x_coordinates[0:4])}...",
            f"y coordinates = {', '.join(self.y_coordinates[0:4])}...")
        return custom_print
@dataclass
class specxploreImportingPipeline ():
    """ 
    Class interface for specXplore importing pipeline functions & storing intermediate data structures. Run pipeline
    methods in the following order with corresponding arguments for constructing a specxplore_session data object:

    spectra_filepath = ... # USER INPUT
    model_directory = ... # USER INPUT

    pipeline = specXploreImportingPipeline(spectra_filepath)
    pipeline.attach_spectral_data_from_file()
    pipeline.run_spectral_processing()
    pipeline.run_spectral_similarity_computationsmodel_directory(model_directory)
    pipeline.run_ms2query(model_directory)
    pipeline.run_and_attach_tsne_grid() --> prints tsne tuning output for selection of perplexity
    pipeline.run_and_attach_kmedoid_grid() --> prints kmedoid tuning output for selection of k_values

    selected_kmedoid_ilocs = [...] # USER INPUT: one or more kmedoid grid ilocs
    selected_tsne_iloc = ... # USER INPUT: one tsne_grid iloc
    pipeline.select_tsne_settings(selected_tsne_iloc)
    pipeline.select_kmedoid_settings(selected_kmedoid_ilocs)

    optional prior to export:
    pipeline.attach_metadata_from_data_frame() # USER INPUT: provide any tabular data to be included in metadata
    pipeline.attach_classes_from_data_frame() # USER INPUT: provide any classification data to be visually included 

    pipeline.export_specxplore_session_data()
    """
    # All default states set to none. Use attach data to construct the actual data object.
    # MSMS spectral data. The core underlying data of specXplore.
    spectra_matchms : Union[List[matchms.Spectrum], None] = None
    spectra_specxplore : Union[List[Spectrum], None] = None
    # Metadata separated into classification style & generic
    classification_table : Union[pd.DataFrame, None] = None
    metadata_table : Union[pd.DataFrame, None] = None
    highlight_table : Union[pd.DataFrame, None] = None # contains "highlight_bool" column for each feature_id
    # Pairwise Similarity Matrices
    primary_score : Union[np.ndarray, None] = None
    secondary_score : Union[np.ndarray, None] = None
    tertiary_score : Union[np.ndarray, None] = None
    # Default specXplore scores implemented
    score_names : List[str] = field(default_factory = lambda: ["ms2deepscore", "spec2vec", "modified-cosine"]) 
    # Embedding grid & coordinates
    tsne_grid : Union[List[TsneGridEntry], None] = None
    tsne_coordinates_table : Union[pd.DataFrame, None] = None
    # K-medoid classification grid
    kmedoid_grid : Union[List[KmedoidGridEntry], None] = None
    # Explicit Booleans to track pipeline steps completed successfully. 
    _spectral_data_loading_complete : bool = False
    _spectral_processing_complete : bool = False
    _add_similarities_complete : bool = False

    # settings used dictionary, initialized as empty
    _settings_used : dict = field(default_factory= lambda: {})

    def attach_spectra_from_file(self, filepath : str) -> None:
        """ 
        Loads and attaches spectra from provided filepath (pointing to compatible .mgf file). Does not run any pre-
        processing. While the function does not check spectral data integrity or performs any filtering, it does make 
        sure that unique feature identifiers are available for all spectra provided.

        Parameters
        filepath : str pointing to a .mgf or .MGF formatted file containing the spectral data. Must have a 
            feature_id entry.
        Returns
        Attaches spectrum_matchms to pipeline instance. Returns None.
        """
        assert isinstance(filepath, str), f"Error: expected filepath to be string but received {type(filepath)}"
        assert os.path.isfile(filepath), "Error: supplied filepath does not point to existing file."
        spectra_matchms = list(matchms.importing.load_from_mgf(filepath))
        _check_spectrum_information_availability(spectra_matchms)
        _ = _extract_feature_ids_from_spectra(spectra_matchms) # loads feature_ids to check uniqueness of every entry
        self.spectra_matchms = spectra_matchms
        self._spectral_data_loading_complete = True
        return None
    def run_spectral_processing(
        self, 
        force : bool = False, 
        **kwargs) -> None:
        """ Runs optional but recommended spectral processing on list of matchms spectra.

        Parameters
            force : bool that defaults to false and prevents the overwriting on previous spectral processing. This is a
                safety step to avoid downstream steps becoming incompatible with the spectral data. Set to true or 
                re-initialize pipeline to rerun spectral processing and rerun any downstream processes.
            **kwargs For optional processing arguments refer to documentation of _generate_processed_spectra()
        
        Return
            Attaches processed spectra_matchms to self. Returns None.
        """

        # apply basic matchms to avoid processing pipeline crashes due to incompatibilities
        assert self.spectra_matchms is not None, (
            "Error: Spectral Processing can only be done if spectral data available."
        )
        if force is False:
            assert self._spectral_processing_complete is False, (
                "Error: spectral processing was already applied. Re-applying may lead to processing errors unless all "
                "subsequent steps are re-run as well! To force a rerun, use force = True or re-initalize the pipeline"
                "instance"
            )
        processed_spectra = _generate_processed_spectra(self.spectra_matchms, **kwargs)
        self.spectra_matchms = processed_spectra
        self._spectral_processing_complete = True
        self._attach_settings_used(**kwargs)
        return None
    def run_spectral_similarity_computations(self, model_directory_path : str = None, force : bool = False):
        """ Runs and attaches spectral similarity measures using self.spectra """
        assert os.path.isdir(model_directory_path), "model directory path must point to an existing directory!"
        if force is False:
            assert (self._add_similarities_complete is not True), (
                "Error: Similarities were already set. To replace existing scores set Force to True or restart "
                "the pipeline."
            )
        self.primary_score = _compute_similarities_ms2ds(self.spectra_matchms, model_directory_path)
        self.secondary_score = _compute_similarities_cosine(self.spectra_matchms, cosine_type="ModifiedCosine")
        self.tertiary_score = _compute_similarities_s2v(self.spectra_matchms, model_directory_path)
        self._add_similarities_complete = True
        self._spectral_processing_complete = True # similarity matrices were computed, this step is skipped and locked
        self._attach_settings_used(score_names = self.score_names)
        return None
    def attach_spectral_similarity_arrays(
        self, 
        primary_score, 
        secondary_score, 
        tertiary_score, 
        score_names : List[str] = ["primary", "secondary", "tertiary"], 
        verbose : bool = True) -> None:
        """ Attaches spectral similarity array computed elsewhere & checks compatibility with spectra. """
        n_spectra = len(self.spectra_matchms)
        _assert_similarity_matrix(primary_score, n_spectra)
        _assert_similarity_matrix(secondary_score, n_spectra)
        _assert_similarity_matrix(tertiary_score, n_spectra)
        if verbose is True:
            warnings.warn((
                "Beware of order misalignment: attach_spectral_similarity_arrays() assumes that the provided score "
                "matrices align in iloc to feature_id mapping with the spectra list provided. If spectra have been "
                "reordered in any way this mapping may not hold!"
            ))
        self.primary_score = primary_score
        self.secondary_score = secondary_score
        self.tertiary_score = tertiary_score
        self.score_names = score_names
        self._add_similarities_complete = True
        self._spectral_processing_complete = True # similarity matrices were computed, this step is skipped and locked
        self._attach_settings_used(score_names = self.score_names)
        return None
    def attach_ms2query_results(self, results_filepath : str):
        """ Function to attach existing ms2query results. Beware: ms2query works with query number identifiers that are
        the equivalent of specxplore spectrum_iloc +1. Making use of attach requires the matchms spectra list to be 
        equivalent to self.spectra_matchms (same spectra, same processing, same order). """
        assert os.path.isfile(results_filepath), (
            f"Error: no file found in provided results_filepath = {results_filepath}"
        )
        self._load_ms2query_results(results_filepath)
        self._attach_settings_used(ms2query_settings = "attached-from-file")
        return None
    def run_ms2query(
        self, 
        model_directory_path : str, 
        results_filepath : str = None, 
        force : bool = False, 
        ms2querySettings : SettingsRunMS2Query = None) -> None:
        """ Runs ms2query for all spectra in set and produces an output file in output/ms2query_results.csv if no
        results_filepath is provided. 
        
        Parameters
            model_directory_path : str path pointing to model and library folder directory
            results_filepath : str optional path to results filename, defaults to output/ms2query_results.csv
            force : bool defaults to false and prevents the running of ms2query when a ms2query file already exists.
            ms2querySettings : SettingsRunMS2Query defaults to None. For control over ms2query settings please refer to 
                ms2query documentation.
        Returns
            Creates ms2query output csv file. Attaches ms2query results to metadata and class tables.
        """
        if results_filepath is None:
            results_filepath = os.path.join("output", "ms2query_results.csv")
        assert not (os.path.isfile(results_filepath) and force is False), (
            f"MS2Query Results with filepath {results_filepath} already exist. Specify alternative filepath, " 
            "delete or rename existing file, or rerun run_ms2query with force == True to automatically overwrite the"
            "existing file.")
        if os.path.exists(results_filepath) and force is True:
            os.remove(results_filepath)
        ms2library = create_library_object_from_one_dir(model_directory_path)
        ms2library.analog_search_store_in_csv(self.spectra_matchms, results_filepath, ms2querySettings)
        self._load_ms2query_results(results_filepath)
        if ms2querySettings is None:
            self._attach_settings_used(ms2query_settings = "default-settings")
        else:
            self._attach_settings_used(ms2query_settings = "non-default-settings")
        return None
    def run_and_attach_tsne_grid(self, perplexity_values : List[int] = [10, 30, 50]) -> None:
        """ Run the t-SNE grid & attach the results to pipeline instance. """
        # Subset perplexity values
        perplexity_values = [perplexity for perplexity in perplexity_values if perplexity < len(self.spectra_matchms)]
        _check_perplexities(perplexity_values, len(self.spectra_matchms))
        distance_matrix = _convert_similarity_to_distance(self.primary_score)
        self.tsne_grid = _run_tsne_grid(distance_matrix, perplexity_values)
        _print_tsne_grid(self.tsne_grid)
        return None
    def select_tsne_settings(self, iloc : int):
        """ Select particular t-SNE coordinate setting using entry iloc. """
        # check iloc valid
        assert self.tsne_grid is not None, (
            "Error: tsne_grid is None. Please run 'run_and_attach_tsne_grid' before selecting a value."
        )
        assert iloc in [x for x in range(0, len(self.tsne_grid))], (
            f"Error: must provide iloc in range of tsne grid 0 to {len(self.tsne_grid)}"
        )
        tsne_coordinates_table = _construct_init_table(self.spectra_matchms)
        tsne_coordinates_table["x"] = self.tsne_grid[iloc].x_coordinates
        tsne_coordinates_table["y"] = self.tsne_grid[iloc].y_coordinates
        self.tsne_coordinates_table = tsne_coordinates_table
        self._attach_settings_used(tsne_perplexity = self.tsne_grid[iloc].perplexity)
        return None
    def plot_selected_embedding(self) -> None:
        data = self.tsne_coordinates_table
        fig  = plotly.express.scatter(
            data_frame= data, x = "x", y = "y", hover_data=["spectrum_iloc", "feature_id"],
            width=800, height=800
        )
        fig.show()
        return None
    def run_and_attach_kmedoid_grid(self, k_values : List[int] = [10, 30, 50]):
        """ Run the k-medoid grid & attach the results to pipeline instance.

        Parameters
            k_values : List[int] of number of clusters to optimize for.
        Returns
            Attached kmedoid grid to self. Returns None.
         """
        # Subset k_values
        k_values = [value for value in k_values if value < len(self.spectra_matchms)]
        _check_perplexities(k_values, len(self.spectra_matchms))
        distance_matrix = _convert_similarity_to_distance(self.primary_score)
        self.kmedoid_grid = _run_kmedoid_grid(distance_matrix, k_values)
        _print_kmedoid_grid(self.kmedoid_grid)
        return None
    def select_kmedoid_settings(self, ilocs : Union[int, List[int]]):
        """ Select and attach particular k-medoid clustering assignments using entry iloc or list of ilocs. """
        # assert input is int or list of int
        assert isinstance(ilocs, list) or isinstance(ilocs, int), (
            "Unsupported input type, ilocs must be int or list of int!"
        )
        valid_ilocs = set([iloc for iloc in range(0, len(self.kmedoid_grid))])
        if isinstance(ilocs, list):
            for entry in ilocs:
                assert isinstance(entry, int), "Non-int entry found in ilocs list while all entries must be int!"
                assert entry in valid_ilocs, (
                    "Error: iloc provided not in range of valid ilocs for kmedoid grid! Values must be in set: "
                    f"{valid_ilocs}"
                )
        # Make sure an initiated classification_table is available
        if self.classification_table is None:
            self.classification_table = _construct_init_table(self.spectra_matchms)
        # Turn to list for selection handling
        if type(ilocs) == int:
            ilocs = [ilocs]
        # Initialize classes table with only feature_id
        kmedoid_classes = _construct_init_table(self.spectra_matchms) 
        kmedoid_classes.drop('spectrum_iloc', axis=1, inplace=True)
        # Add cluster assignments in string format
        selected_k = []
        for selection_iloc in ilocs:
            cluster_name = f"k_{self.kmedoid_grid[selection_iloc].k}"
            selected_k.append(self.kmedoid_grid[selection_iloc].k)
            cluster_assignments = [f"cluster_{elem}" for elem in self.kmedoid_grid[selection_iloc].cluster_assignments]
            kmedoid_classes.insert(loc=0, column= cluster_name, value = cluster_assignments)
        # remove spectrum_iloc
        self.classification_table = _attach_columns_via_feature_id(self.classification_table, kmedoid_classes)
        self._attach_settings_used(k_medoid_k_values = selected_k)
        return None
    def attach_metadata_from_data_frame(self, addon_data : pd.DataFrame) -> None:
        """ Attach additional metadata contained within pd.DataFrame to existing metadata via feature_id overlap. 
        
        Parameters
            addon_data : pd.DataFrame with feature_id column (subset/superset of those in spectral_data) and additional 
                columns to be included.
        Returns 
            Attaches new data to metadatTable. Returns None.
        """
        if self.metadata_table is None:
            self.metadata_table = _construct_init_table(self.spectra_matchms)
        self.metadata_table = _attach_columns_via_feature_id(
            self.metadata_table, 
            addon_data
        )
        return None
    def attach_classes_from_data_frame(self, addon_data : pd.DataFrame) -> None:
        """ 
        Attach additional classdata contained within pd.DataFrame to existing class_table via feature_id overlap. 

        Parameters
            addon_data : pd.DataFrame with feature_id column (subset or superset of those in spectral_data) and 
                additional columns to be included. Note that additional data should be suitable categorical data for
                highlighting purposes within the specXplore interactive dashboard.
        Returns 
            Attaches new data to metadatTable. Returns None.
        """
        # Class table may not have been initiated
        if self.classification_table is None:
            self.classification_table = _construct_init_table(self.spectra_matchms)
        self.classification_table = _remove_white_space_from_df(
            _attach_columns_via_feature_id(self.classification_table, addon_data)
        )
        return None
    def export_specxplore_session_data(self, filepath : str = None, force : bool = False):
        # if filepath is None, construct filepath automatically
        if filepath is None:
            filepath = os.path.join("output", "specxplore_session_data.pickle")
        assert not (os.path.isfile(filepath) and force is False), (
            f"File with filepath {filepath} already exists. Specify alternative filepath, " 
            "delete or rename existing file, or rerun export_specxplore_session_data with force == True to "
            "automatically overwrite the existing file.")
        if os.path.exists(filepath) and force is True:
            os.remove(filepath)

        if self.highlight_table is None: 
            self.attach_feature_highlights() # will initiate feature highlight to all non-highlighted
        
        # convert matchms spectra to format used by dashboard code
        self.spectra_specxplore = _convert_matchms_spectra_to_specxplore_spectra(self.spectra_matchms)
        
        # Check for all data available to run session data constructor
        assert self.spectra_specxplore is not None, "Error: spectra_specxplore required but not available."
        assert self.tsne_coordinates_table is not None, "Error: tsne_coordinates_table required but not available."
        assert self.classification_table is not None, "Error: classification_table required but not available."
        assert self.metadata_table is not None, "Error: metadata_table required but not available."
        assert self.primary_score is not None, "Error: primary_score required but not available."
        assert self.secondary_score is not None, "Error: secondary_score required but not available."
        assert self.tertiary_score is not None, "Error: tertiary_score required but not available."
        assert self.score_names is not None, "Error: score_names required but not available."
        assert self._settings_used is not None, "Error: score_names required but not available."
        # run session data constructor

        session_data = SpecxploreSessionData(
            self.spectra_specxplore, self.tsne_coordinates_table, self.classification_table, self.metadata_table, 
            self.primary_score, self.secondary_score, self.tertiary_score, self.score_names, self.highlight_table,
            self._settings_used
        )
        # pickle session_data object
        with open(filepath, 'wb') as file:
            pickle.dump(session_data, file)
        return None
    def attach_feature_highlights(self, feature_ids : List[str] = []) -> None:
        """ Function attaches special interest feature_ids to be highlighted in specXplore. """
        assert isinstance(feature_ids, list), "Error: provided feature_ids must be stored in list."
        for elem in feature_ids:
            assert isinstance(elem, str), f"Error: provided feature_id must be type str but is type {type(elem)}"
        if self.highlight_table is None:
            self.highlight_table = _construct_init_table(self.spectra_matchms)
        self.highlight_table["highlight_bool"] = [
            True if elem in feature_ids else False for elem in self.highlight_table["feature_id"]
        ]
        return None
    def _load_ms2query_results(self, filepath : str) -> None:
        """ Loads ms2query data corresponding to the run of run_ms2query on self.spectra_matchms and attached results
        to metadata and class tables.
        Returns 
        """
        assert os.path.isfile(filepath), "Error: filepath does not point to existing file!"
        ms2query_annotation_table = pd.read_csv(filepath)

        # Create a mapping of feature_id to query_number
        query_number = [iloc for iloc in range(1, len(self.spectra_matchms)+1)]
        feature_ids = _extract_feature_ids_from_spectra(self.spectra_matchms) 
        query_number_to_iloc_to_feature_id_mapping = pd.DataFrame(
            {
                "feature_id": feature_ids, 
                "query_spectrum_nr" : query_number
            }
        )
        # all ms2query results collected into one table
        ms2query_annotation_table = ms2query_annotation_table.merge(
            query_number_to_iloc_to_feature_id_mapping, 
            how = "left", 
            on = "query_spectrum_nr"
        )
        # Rename ms2query feature identifier column and recast it as string type if not already
        ms2query_annotation_table["feature_id"] = ms2query_annotation_table["feature_id"].astype("string")
        # Extract class part
        classification_table = ms2query_annotation_table.loc[:, [
            'cf_superclass', 'cf_class', 'cf_subclass', 'cf_direct_parent', 'npc_class_results', 
            'npc_superclass_results', 'npc_pathway_results', 'feature_id'
            ]
        ]
        # check requires these tables to be initialized to None. Otherwise an attribute error is produced. 
        self.attach_classes_from_data_frame(classification_table)
        self.attach_metadata_from_data_frame(ms2query_annotation_table)

        return None
    def _attach_settings_used(self, **kwargs) -> None:
        """Attaches used settings to settings dictionary. """
        for key, value in kwargs.items():
            if key is not None and value is not None:
                self._settings_used[key] = value
        return None
def _construct_init_table(spectra : List[matchms.Spectrum]) -> pd.DataFrame:
    ''' Creates initialized table for metadata or classification in specXplore. Table is a pandas.DataFrame with
    string and int columns indicating the feature_id, and spectrum_iloc.
    
    Parameters:
        spectra : List[matchms.Spectrum] where each spectrum contains a unique feature_identifier (str).
    Returns:
        pd.DataFrame with feature_id and spectrum_iloc columns (dtypes: string, int64)
    '''
    feature_ids = _extract_feature_ids_from_spectra(spectra)
    spectrum_ilocs = [iloc for iloc in range(0, len(spectra))]

    init_table = pd.DataFrame(data = {"feature_id" : feature_ids,  "spectrum_iloc" : spectrum_ilocs})
    init_table["feature_id"] = init_table["feature_id"].astype("string")
    init_table["spectrum_iloc"] = init_table["spectrum_iloc"].astype("int64")
    return init_table
def _convert_matchms_spectra_to_specxplore_spectra(
    spectra = List[matchms.Spectrum]
    ) -> List[Spectrum]:
    """ Converts list of matchms.Spectrum objects to list of specxplore.importing.Spectrum objects. """
    spectra_converted = [
        Spectrum(
            mass_to_charge_ratios = spec.peaks.mz, 
            precursor_mass_to_charge_ratio = float(spec.get("precursor_mz")), 
            spectrum_iloc = idx, 
            intensities = spec.peaks.intensities, 
            feature_id=spec.get("feature_id")    
        ) 
        for idx, spec 
        in enumerate(spectra)
    ]
    return spectra_converted
def _assert_similarity_matrix(scores : np.ndarray, n_spectra : int) -> None:
    """ Function checks whether similarity matrix corresponds to expected formatting. Aborts code if not. """
    assert (isinstance(scores, np.ndarray)), "Error: input scores must be type np.ndarray."
    assert scores.shape[0] == scores.shape[1] == n_spectra, (
        "Error: score dimensions must be square & correspond to n_spectra"
    )
    assert np.logical_and(scores >= 0, scores <= 1).all(), "Error: all score values must be in range 0 to 1."
    return None
def _check_spectrum_information_availability(spectra : List[matchms.Spectrum]) -> None:
    """ Checks if list of spectral data contains expected entries. Aborts code if not the case. """
    for spectrum in spectra:
        assert isinstance(spectrum, matchms.Spectrum), (
            f"Error: item in loaded spectrum list is not of type matchms.Spectrum!"
        )
        assert spectrum is not None, (
            "Error: None object detected in spectrum list. All spectra must be valid matchms.Spectrum instances."
        )
        assert spectrum.get("feature_id") is not None, (
            "Error: All spectra must have valid feature_idid entries."
        )
        assert spectrum.get("precursor_mz") is not None, (
            "Error: All spectra must have valid precursor_mz value."
        )
    return None
def _extract_feature_ids_from_spectra(spectra : List[matchms.Spectrum]) -> List[str]:
    """ Extract feature ids from list of matchms spectra in string format. """
    # Extract feature ids from matchms spectra. 
    feature_ids = [str(spec.get("feature_id")) for spec in spectra]
    # check feature_id set validity
    assert not feature_ids == [], "Error: no feature ids detected!"
    assert not any(feature_ids) is None, (
        "Error: None type feature ids detected! All spectra must have valid feature_id entry of type string."
    )
    assert not all(feature_ids) is None, (
        "Error: None type feature ids detected! All spectra must have valid feature_id entry of type string."
    )
    assert all(isinstance(x, str) for x in feature_ids), (
        "Error: Non-string feature_ids detected. All feature_ids for spectra must be valid string type."
    )
    assert not (len(feature_ids) > len(set(feature_ids))), (
        "Error: Non-unique (duplicate) feature_ids detected. All feature_ids for spectra must be unique strings."
    )
    return feature_ids
def _check_perplexities(perplexity_values : List[Union[float, int]], max_perplexity : Union[float, int]) -> None:
    """ Function checks whether perplexity values match expected configuration. Aborts if not. """
    assert perplexity_values is not [], (
        "Error: perplexity_values list is empty! This may be a result of post-processing: there must be a "
        "perplexity value below the number of spectra for optimization to work."
    )
    assert isinstance(perplexity_values, list), (
        "Error: perplexity values must be a list. If only running one value, specify input as [value]."
    )
    for perplexity_value in perplexity_values: 
        assert isinstance(perplexity_value, (int, float)) and perplexity_value < max_perplexity, (
            "Error: perplexity values must be numeric (int, float) and smaller than number of features." 
        )
    return None
def _check_k_values(k_values : List[int], max_k : int) -> None:
    """ Function checks whether k values match expected configuration. Aborts if not. """
    assert k_values is not [], (
        "Error: k_values list is empty! This may be a result of post-processing: there must be a "
        "k value below the number of features/spectra for optimization to work."
    )
    assert isinstance(k_values, list), (
        "Error: k_values must be a list. If only running one value, specify input as [value]."
    )
    for k_value in k_values: 
        assert isinstance(k_value, int) and k_value < max_k, (
            "Error: k_value must be numeric (int) and smaller than number of features/spectra." 
        )
    return None
def _run_tsne_grid(
        distance_matrix : np.ndarray,
        perplexity_values : List[int], 
        random_states : Union[List, None] = None
        ) -> List[TsneGridEntry]:
    """ Runs t-SNE embedding routine for every provided perplexity value in perplexity_values list.

    Parameters:
        distance_matrix: An np.ndarray containing pairwise distances.
        perplexity_values: A list of perplexity values to try for t-SNE embedding.
        random_states: None or a list of integers specifying the random state to use for each k-medoid run.
    Returns: 
        A list of TsneGridEntry objects containing grid results. 
    """
    _check_perplexities(perplexity_values, distance_matrix.shape[0])
    if random_states is None:
        random_states = [ 0 for _ in perplexity_values ]
    output_list = []
    for idx, perplexity in enumerate(perplexity_values):
        model = TSNE(
            metric="precomputed", 
            random_state = random_states[idx], 
            init = "random", 
            perplexity = perplexity
        )
        z = model.fit_transform(distance_matrix)
        # Compute embedding quality
        dist_tsne = squareform(pdist(z, 'seuclidean'))
        spearman_score = np.array(spearmanr(distance_matrix.flat, dist_tsne.flat))[0]
        pearson_score = np.array(pearsonr(distance_matrix.flat, dist_tsne.flat))[0]
        output_list.append(
            TsneGridEntry(
                perplexity, 
                z[:,0], 
                z[:,1], 
                pearson_score, 
                spearman_score, 
                random_states[idx]
            )
        )
    return output_list
def _plot_tsne_grid(tsne_list : List[TsneGridEntry]) -> None:
    """ Plots pearson and spearman scores vs perplexity for each entry in list of TsneGridEntry objects. """
    
    pearson_scores = [x.spearman_score for x in tsne_list]
    spearman_scores = [x.pearson_score for x in tsne_list]
    iloc_perplexity = [ f"{x.perplexity} / {iloc}" for iloc, x in enumerate(tsne_list)]

    trace_spearman = go.Scatter(x = iloc_perplexity, y = spearman_scores, name="spearman_score", mode = "markers")
    trace_pearson = go.Scatter(x = iloc_perplexity, y = pearson_scores, name="pearson_score", mode = "markers")
    fig = go.Figure([trace_pearson, trace_spearman])
    fig.update_layout(xaxis_title="Perplexity / iloc", yaxis_title="Score")
    fig.show()
    return None
def _convert_similarity_to_distance(similarity_matrix : np.ndarray) -> np.ndarray:
    """ 
    Converts pairwise similarity matrix to distance matrix with values between 0 and 1. Assumes that the input is a
    similarity matrix with values in range 0 to 1 up to floating point error.
    """
    distance_matrix = 1.- similarity_matrix
    distance_matrix = np.round(distance_matrix, 6) # Round to deal with floating point issues
    distance_matrix = np.clip(distance_matrix, a_min = 0, a_max = 1) # Clip to deal with floating point issues
    return distance_matrix
def _run_kmedoid_grid(
        distance_matrix : np.ndarray, 
        k_values : List[int], 
        random_states : Union[List, None] = None
        ) -> List[KmedoidGridEntry]:
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
    _check_k_values(k_values, max_k = distance_matrix.shape[0])
    for idx, k in enumerate(k_values):
        cluster = KMedoids(
            n_clusters=k, 
            metric='precomputed', 
            random_state=random_states[idx], 
            method = "fasterpam"
        )  
        cluster_assignments = cluster.fit_predict(distance_matrix)
        cluster_assignments_strings = [
            "km_" + str(elem) 
            for elem in cluster_assignments
        ]
        score = silhouette_score(
            X = distance_matrix, 
            labels = cluster_assignments_strings, 
            metric= "precomputed"
        )
        output_list.append(
            KmedoidGridEntry(
                k, 
                cluster_assignments_strings, 
                score, 
                random_states[idx]
            )
        )
    return output_list
def _plot_kmedoid_grid(
        kmedoid_list : List[KmedoidGridEntry]
        ) -> None:
    """ Plots Silhouette Score vs k for each entry in list of KmedoidGridEntry objects. """
    scores = [x.silhouette_score for x in kmedoid_list]
    ks = [f"k = {x.k} / iloc = {iloc}" for iloc, x in enumerate(kmedoid_list)]
    fig = plotly.express.scatter(x = ks, y = scores)
    fig.update_layout(
        xaxis_title="K (Number of Clusters) / iloc", 
        yaxis_title="Silhouette Score"
    )
    fig.show()
    return None
def _print_kmedoid_grid(grid : List[KmedoidGridEntry]) -> None:
    """ Prints all values in kmedoid grid in readable format via pandas conversion """
    kmedoid_results = pd.DataFrame.from_dict(data = grid).loc[
        :, ["k", "silhouette_score", "random_seed_used"]
    ]
    kmedoid_results.insert(loc = 0, column = "iloc", value = [iloc for iloc in range(0, len(grid))])
    print("Kmedoid grid results. Use to inform kmedoid classification selection ilocs.")
    print(kmedoid_results)
    return None
def _print_tsne_grid(grid : List[TsneGridEntry]) -> None:   
    """ Prints all values in tsne grid in readable format via pandas conversion """
    tsne_results = pd.DataFrame.from_dict(data = grid).loc[
        :, ["perplexity", "pearson_score", "spearman_score", "random_seed_used"]
    ]
    tsne_results.insert(loc = 0, column = "iloc", value = [iloc for iloc in range(0, len(grid))])
    print("T-sne grid results. Use to inform t-sne embedding selection.")
    print(tsne_results)
    return None
def _attach_columns_via_feature_id(init_table : pd.DataFrame, addon_data : pd.DataFrame,) -> pd.DataFrame:
    """ Attaches addon_data to data frame via join on 'feature_id'. 
    
    The data frame can be a class table or metadata table and is assumed to be derived from the _construct_init_table() 
    and contain feature_id and spectrum_iloc columns.
    
    Input
        init_table: pandas.DataFrame object with at least a feature_id column (type string)
        addon_data: pandas.DataFrame object with a feature_id column and additional columns to be merged into metadata. 
            Columns can be of any type.
    Output: 
        extended_init_table: pandas.DataFrame with feature_id column and additional columns from addon_data. Any NA 
            values produced are replaced with strings that read: "not available". Any entries are converted to string.
    """

    assert "feature_id" in init_table.columns, "feature_id column must be available in metadata"
    assert "feature_id" in addon_data.columns, "feature_id column must be available in addon_data"
    assert (init_table["feature_id"].dtype 
            == addon_data["feature_id"].dtype 
            == 'string'), (
        "feature_id column must be of the same type."
    )
    extended_init_table = copy.deepcopy(init_table)
    extended_init_table = extended_init_table.merge(
        addon_data.loc[:, ~addon_data.columns.isin(['spectrum_iloc'])],
        left_on = "feature_id", 
        right_on = "feature_id", 
        how = "left"
    )
    extended_init_table.reset_index(inplace=True, drop=True)
    extended_init_table = extended_init_table.astype('string')
    extended_init_table = extended_init_table.replace(
        to_replace=np.nan, 
        value = "not_available"
    )
    return extended_init_table
def _remove_white_space_from_df(input_df : pd.DataFrame) -> pd.DataFrame:
    ''' Removes whitespace from all entries in input_df. 
    
    White space removal is essential for accurate chemical classification parsing in node highlighting of specXplore.
    '''
    output_df = copy.deepcopy(input_df)
    output_df = input_df.replace(to_replace = " ", value = "_", regex=True)
    output_df = output_df.replace(to_replace = ";", value = "--", regex=True)
    output_df = output_df.replace(to_replace = ",", value = "", regex=True)
    output_df = output_df.replace(to_replace = ",", value = "", regex=True)
    # remove any remaining non numeric or letter
    output_df = output_df.replace(to_replace = '\W', value = '', regex = True)  
    return output_df
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
def _compute_similarities_ms2ds(spectrum_list:List[matchms.Spectrum], model_path:str) -> np.ndarray:
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
    scores_matchms = calculate_scores(
        spectrum_list, spectrum_list, similarity_measure, is_symmetric=True, array_type="numpy"
    )
    scores_ndarray = scores_matchms.to_array()
    scores_ndarray = np.clip(scores_ndarray, a_min = 0, a_max = 1) # Clip to deal with floating point issues
    return scores_ndarray
def _compute_similarities_s2v(spectrum_list:List[matchms.Spectrum], model_path:str) -> np.ndarray:
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
    scores_matchms = calculate_scores(
        spectrum_list, spectrum_list, similarity_measure, is_symmetric=True, array_type="numpy")
    scores_ndarray = scores_matchms.to_array()
    # Dev Note: spec2vec scores appear to be in range -1 to 1 (with floating point errors). For specXplore they must
    # be in range 0 to 1. Apply linear scaling to put spec2vec scores into range between 0 and 1 
    # (from their original -1 to 1 range)
    scores_ndarray = (scores_ndarray + 1) / 2 # linear scaling
    scores_ndarray = np.clip(scores_ndarray, a_min = 0, a_max = 1) # Clip to deal with floating point issues
    # Dev Note: note that numeric distance between lowest and highest number in the score vector will be half of the
    # original value in this approach, e.g. the distance from -1 to 1 is 2, while in the new space 0 to 1 distance is
    # at most 1. This implies a subtle difference in interpreting this score.
    return scores_ndarray
def _compute_similarities_cosine(spectrum_list:List[matchms.Spectrum], cosine_type : str = "ModifiedCosine"):
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
    tmp = calculate_scores(spectrum_list, spectrum_list, similarity_measure, is_symmetric=True, array_type = "numpy")
    scores = _extract_similarity_scores_from_matchms_cosine_array(tmp.to_array())
    scores = np.clip(scores, a_min = 0, a_max = 1) 
    return scores
def _extract_similarity_scores_from_matchms_cosine_array(tuple_array : np.ndarray) -> np.ndarray:
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
def _generate_processed_spectra(
        input_spectra : List[matchms.Spectrum],
        minimum_number_of_peaks : int = 5,
        maximum_number_of_peaks : int = 200,
        max_mz : float = 1000,
        min_mz :float = 0,
        verbose : bool = True
        ) -> List[matchms.Spectrum]:
    ''' 
    Applies spectral data pre-processing to copy of input_spectra. Runs a matchms pipeline that ensures the following
    conditions in step-wise fashion:

        1. peak intensities are normalized
        2. peaks are filtered to be in range 0 to 1000 m/z in line with model expectations. 
        3. spectra are excluded if they have less than minimum_number_of_peaks, and if the spectrum has
        more than maximum_number_of_peaks the lowest intensity peaks exceeding the limits are removed. 
        4. precursor_mz metadata is in expected format
        5. Any None entries from the list of spectra are removed (None types are produced if spectra are are empty)
    
    This is a minimal pre-processing pipeline based on matchms. More extensive pre-processing may be done by the user
    using functionalities in matchms.

    Parameters:
        input_spectra : A List[matchms.Spectrum] as loaded into python via matchms. 
        minimum_number_of_peaks : An int specifying the minimum number of peaks per spectrum. 
            Spectra with less peaks are removed.
        maximum_number_of_peaks : An int specifying the maximum number of peaks per spectrum. If the 
            spectrum exceeds this number of peaks, the exceeding number of peaks are removed starting with the lowest
            intensity ones.
        max_mz : A float specifying the maximum mass to charge ratio for peaks in a spectrum. Peaks exceeding this value
            are removed. 
        min_mz : A float specifying the minimum mass to charge ratio for peaks in a spectrum. Peaks below this value
            are removed. 
        verbose : Boolean specifying whether processing effects on spectrum list length are printed to console or not.
            Defaults to true.
    Returns:
        List[matchms.Spectrum] with filters applied. Note that the function will abort if pre-processing leads to empty
        list.
    '''
    # TODO: implement checks; this function would be safer for end users if strict boundary checking was applied on 
    # input parameters.
    if verbose:
        print("Number of spectra prior to filtering: ", len(input_spectra))
    # Normalize intensities, important for similarity measures
    processed_spectra = copy.deepcopy(input_spectra)
    processed_spectra = [matchms.filtering.normalize_intensities(spec) for spec in processed_spectra]
    processed_spectra = [
        matchms.filtering.select_by_mz(spec, mz_from = min_mz, mz_to = max_mz) for spec in processed_spectra
    ]
    # Clean spectra by remove very low intensity fragments, noise removal
    processed_spectra = [
        matchms.filtering.reduce_to_number_of_peaks(
            spec, n_required = minimum_number_of_peaks, n_max= maximum_number_of_peaks) 
        for spec in processed_spectra
    ]
    # Add precursor mz values to matchms spectrum entry
    processed_spectra = [matchms.filtering.add_precursor_mz(spec)  for spec in processed_spectra]
    # remove none entries in list (no valid spectrum returned)
    processed_spectra = [spec for spec in processed_spectra if spec is not None]
    # Assess whether processing results in legitimate output
    assert processed_spectra != [], "Error: no spectra left after applying processing!"
    if verbose:
        print("Number of spectra after to filtering: ", len(processed_spectra))
    return processed_spectra
