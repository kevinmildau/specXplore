# code adapted from ms2query clean_and_filter_spectra.py and ms2query utils.py

from typing import List, Tuple
import matchms.filtering as msfilters
from tqdm import tqdm
from matchms import Spectrum
from matchms.utils import is_valid_inchi, is_valid_inchikey, is_valid_smiles
from matchms.typing import SpectrumType
from matchms.logging_functions import set_matchms_logger_level
from matchmsextras.pubchem_lookup import pubchem_metadata_lookup
from spec2vec import SpectrumDocument


def clean_metadata(spectrum: Spectrum) -> Spectrum:
    spectrum = msfilters.default_filters(spectrum)
    spectrum = msfilters.add_retention_index(spectrum)
    spectrum = msfilters.add_retention_time(spectrum)
    spectrum = msfilters.require_precursor_mz(spectrum)
    return spectrum


def normalize_and_filter_peaks(spectrum: Spectrum) -> Spectrum:
    """Spectrum is normalized and filtered"""
    spectrum = msfilters.normalize_intensities(spectrum)
    spectrum = msfilters.select_by_relative_intensity(spectrum, 0.001, 1)
    spectrum = msfilters.select_by_mz(spectrum, mz_from=0.0, mz_to=1000.0)
    spectrum = msfilters.reduce_to_number_of_peaks(spectrum, n_max=500)
    spectrum = msfilters.require_minimum_number_of_peaks(spectrum, n_required=3)
    return spectrum


def create_spectrum_documents(query_spectra: List[Spectrum],
                              progress_bar: bool = False,
                              nr_of_decimals: int = 2
                              ) -> List[SpectrumDocument]:
    """Transforms list of Spectrum to List of SpectrumDocument

    Args
    ------
    query_spectra:
        List of Spectrum objects that are transformed to SpectrumDocument
    progress_bar:
        When true a progress bar is shown. Default = False
    nr_of_decimals:
        The number of decimals used for binning the peaks.
    """
    spectrum_documents = []
    for spectrum in tqdm(query_spectra,
                         desc="Converting Spectrum to Spectrum_document",
                         disable=not progress_bar):
        spectrum = msfilters.add_losses(spectrum,
                                        loss_mz_from=5.0,
                                        loss_mz_to=200.0)
        spectrum_documents.append(SpectrumDocument(
            spectrum,
            n_decimals=nr_of_decimals))
    return spectrum_documents


def harmonize_annotation(spectrum: Spectrum,
                         do_pubchem_lookup) -> Spectrum:
    set_matchms_logger_level("CRITICAL")
    # Here, undefiend entries will be harmonized (instead of having a huge variation of None,"", "N/A" etc.)
    spectrum = msfilters.harmonize_undefined_inchikey(spectrum)
    spectrum = msfilters.harmonize_undefined_inchi(spectrum)
    spectrum = msfilters.harmonize_undefined_smiles(spectrum)

    # The repair_inchi_inchikey_smiles function will correct misplaced metadata
    # (e.g. inchikeys entered as inchi etc.) and harmonize the entry strings.
    spectrum = msfilters.repair_inchi_inchikey_smiles(spectrum)

    # Where possible (and necessary, i.e. missing): Convert between smiles, inchi, inchikey to complete metadata.
    # This is done using functions from rdkit.
    spectrum = msfilters.derive_inchi_from_smiles(spectrum)
    spectrum = msfilters.derive_smiles_from_inchi(spectrum)
    spectrum = msfilters.derive_inchikey_from_inchi(spectrum)

    # Adding parent mass is relevant for pubchem lookup
    if do_pubchem_lookup:
        if not check_fully_annotated(spectrum):
            spectrum = msfilters.add_parent_mass(spectrum, estimate_from_adduct=True)
            spectrum = pubchem_metadata_lookup(spectrum,
                                               mass_tolerance=2.0,
                                               allowed_differences=[(18.03, 0.01),
                                                                    (18.01, 0.01)],
                                               name_search_depth=15)
    return spectrum


def remove_wrong_ion_modes(spectra, ion_mode_to_keep):
    assert ion_mode_to_keep in {"positive", "negative"}, "ion_mode should be set to 'positive' or 'negative'"
    spectra_to_keep = []
    for spectrum in tqdm(spectra,
                         desc=f"Selecting {ion_mode_to_keep} mode spectra"):
        if spectrum is not None:
            if spectrum.get("ionmode") == ion_mode_to_keep:
                spectra_to_keep.append(spectrum)
    print(f"From {len(spectra)} spectra, "
          f"{len(spectra) - len(spectra_to_keep)} are removed since they are not in {ion_mode_to_keep} mode")
    return spectra_to_keep


def check_fully_annotated(spectrum: Spectrum) -> bool:
    if not is_valid_smiles(spectrum.get("smiles")):
        return False
    if not is_valid_inchikey(spectrum.get("inchikey")):
        return False
    if not is_valid_inchi(spectrum.get("inchi")):
        return False
    spectrum = msfilters.require_precursor_mz(spectrum)
    if spectrum is None:
        return False
    return True


def split_annotated_spectra(spectra: List[Spectrum]) -> Tuple[List[Spectrum], List[Spectrum]]:
    fully_annotated_spectra = []
    not_fully_annotated_spectra = []
    for spectrum in tqdm(spectra,
                         desc="Splitting annotated and unannotated spectra"):
        fully_annotated = check_fully_annotated(spectrum)
        if fully_annotated:
            fully_annotated_spectra.append(spectrum)
        else:
            not_fully_annotated_spectra.append(spectrum)
    print(f"From {len(spectra)} spectra, "
          f"{len(spectra) - len(fully_annotated_spectra)} are removed since they are not fully annotated")
    return fully_annotated_spectra, not_fully_annotated_spectra


def normalize_and_filter_peaks_multiple_spectra(spectrum_list: List[SpectrumType],
                                                progress_bar: bool = False
                                                ) -> List[SpectrumType]:
    """Preprocesses all spectra and removes None values

    Args:
    ------
    spectrum_list:
        List of spectra that should be preprocessed.
    progress_bar:
        If true a progress bar will be shown.
    """
    for i, spectrum in enumerate(
            tqdm(spectrum_list,
                 desc="Preprocessing spectra",
                 disable=not progress_bar)):
        processed_spectrum = normalize_and_filter_peaks(spectrum)
        spectrum_list[i] = processed_spectrum

    # Remove None values
    return [spectrum for spectrum in spectrum_list if spectrum]


def clean_normalize_and_split_annotated_spectra(spectra: List[Spectrum],
                                                ion_mode_to_keep,
                                                do_pubchem_lookup=True) -> Tuple[List[Spectrum], List[Spectrum]]:
    spectra = [clean_metadata(s) for s in tqdm(spectra, desc="Cleaning metadata")]
    spectra = remove_wrong_ion_modes(spectra, ion_mode_to_keep)
    spectra = [harmonize_annotation(s, do_pubchem_lookup) for s in tqdm(spectra, desc="Harmonizing annotations")]
    spectra = normalize_and_filter_peaks_multiple_spectra(spectra, progress_bar=True)
    annotated_spectra, unannotated_spectra = split_annotated_spectra(spectra)
    # Both annotated and unannotated spectra are returned to make it possible to still use them for Spec2Vec training
    return annotated_spectra, unannotated_spectra


    import os
import sys
import json
from typing import List, Union
import numpy as np
import pandas as pd
from matchms import importing
from spec2vec.Spec2Vec import Spectrum


if sys.version_info < (3, 8):
    import pickle5 as pickle
else:
    import pickle


def load_ms2query_model(ms2query_model_file_name):
    """Loads in a MS2Query model

    a .pickle file is loaded like a ranadom forest from sklearn

    ms2query_model_file_name:
        The file name of the ms2query model
    """
    assert os.path.exists(ms2query_model_file_name), "MS2Query model file name does not exist"
    file_extension = os.path.splitext(ms2query_model_file_name)[1].lower()

    if file_extension == ".pickle":
        return load_pickled_file(ms2query_model_file_name)

    raise ValueError("The MS2Query model file is expected to end on .pickle")


def save_pickled_file(obj, filename: str):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, "wb") as f:
        pickle.dump(obj, f)


def save_json_file(data, filename):
    assert not os.path.exists(filename), "File already exists"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f)


def load_json_file(filename):
    with open(filename, "r", encoding='utf-8') as f:
        data = json.load(f)
    return data


def load_pickled_file(filename: str):
    with open(filename, 'rb') as file:
        loaded_object = pickle.load(file)
    return loaded_object


def load_matchms_spectrum_objects_from_file(file_name
                                            ) -> Union[List[Spectrum], None]:
    """Loads spectra from your spectrum file into memory as matchms Spectrum object

    The following file extensions can be loaded in with this function:
    "mzML", "json", "mgf", "msp", "mzxml", "usi" and "pickle".
    A pickled file is expected to directly contain a list of matchms spectrum objects.

    Args:
    -----
    file_name:
        Path to file containing spectra, with file extension "mzML", "json", "mgf", "msp",
        "mzxml", "usi" or "pickle"
    """
    assert os.path.exists(file_name), f"The specified file: {file_name} does not exists"

    file_extension = os.path.splitext(file_name)[1].lower()
    if file_extension == ".mzml":
        return list(importing.load_from_mzml(file_name))
    if file_extension == ".json":
        return list(importing.load_from_json(file_name))
    if file_extension == ".mgf":
        return list(importing.load_from_mgf(file_name))
    if file_extension == ".msp":
        return list(importing.load_from_msp(file_name))
    if file_extension == ".mzxml":
        return list(importing.load_from_mzxml(file_name))
    if file_extension == ".usi":
        return list(importing.load_from_usi(file_name))
    if file_extension == ".pickle":
        spectra = load_pickled_file(file_name)
        assert isinstance(spectra, list), "Expected list of spectra"
        assert isinstance(spectra[0], Spectrum), "Expected list of spectra"
        return spectra
    assert False, f"File extension of file: {file_name} is not recognized"


def add_unknown_charges_to_spectra(spectrum_list: List[Spectrum],
                                   charge_to_use: int = 1,
                                   change_all_spectra: bool = False) -> List[Spectrum]:
    """Adds charges to spectra when no charge is known

    The charge is important to calculate the parent mass from the mz_precursor
    This function is not important anymore, since the switch to using mz_precursor

    Args:
    ------
    spectrum_list:
        List of spectra
    charge_to_use:
        The charge set when no charge is known. Default = 1
    change_all_spectra:
        If True the charge of all spectra is set to this value. If False only the spectra that do not have a specified
        charge will be changed.
    """
    if change_all_spectra:
        for spectrum in spectrum_list:
            spectrum.set("charge", charge_to_use)
    else:
        for spectrum in spectrum_list:
            if spectrum.get("charge") is None:
                spectrum.set("charge", charge_to_use)
    return spectrum_list


def get_classifier_from_csv_file(classifier_file_name: str,
                                 list_of_inchikeys: List[str]):
    """Returns a dataframe with the classifiers for a selection of inchikeys

    Args:
    ------
    csv_file_name:
        File name of text file with tap separated columns, with classifier
        information.
    list_of_inchikeys:
        list with the first 14 letters of inchikeys, that are selected from
        the classifier file.
    """
    assert os.path.isfile(classifier_file_name), \
        f"The given classifier csv file does not exist: {classifier_file_name}"
    classifiers_df = pd.read_csv(classifier_file_name, sep="\t")
    classifiers_df.rename(columns={"inchi_key": "inchikey"}, inplace=True)
    columns_to_keep = ["inchikey"] + column_names_for_output(False, True)
    list_of_classifiers = []
    for inchikey in list_of_inchikeys:
        classifiers = classifiers_df.loc[
            classifiers_df["inchikey"].str.startswith(inchikey)]
        if classifiers.empty:
            list_of_classifiers.append(pd.DataFrame(np.array(
                [[inchikey] + [np.nan] * (len(columns_to_keep) - 1)]),
                columns=columns_to_keep))
        else:
            classifiers = classifiers[columns_to_keep].iloc[:1]

            list_of_classifiers.append(classifiers)
    if len(list_of_classifiers) == 0:
        results = pd.DataFrame(columns=columns_to_keep)
    else:
        results = pd.concat(list_of_classifiers, axis=0, ignore_index=True)

    results["inchikey"] = list_of_inchikeys
    return results


def column_names_for_output(return_non_classifier_columns: bool,
                            return_classifier_columns: bool,
                            additional_metadata_columns: List[str] = None,
                            additional_ms2query_score_columns: List[str] = None) -> List[str]:
    """Returns the column names for the output of results table

    This is used by the functions MS2Library.analog_search_store_in_csv, ResultsTable.export_to_dataframe
    and get_classifier_from_csv_file. The column names are used to select which data is added from the ResultsTable to
    the dataframe and the order of these columns is also used as order for the columns in this dataframe.

    Args:
    ------
    return_standard_columns:
        If true all columns are returned that do not belong to the classifier_columns. This always includes the
        standard_columns and if if additional_metadata_columns or additional_ms2query_score_columns is specified these
        are appended.
        If return_classifier_columns is True, the classifier_columns are also appended to the columns list.
    return_classifier_columns:
        If true the classifier columns appended. If return_standard_columns is false and return_classifier_columns is
        True, only the classifier columns are returned.
    additional_metadata_columns:
        These columns are appended to the standard columns and returned when return_non_classifier_columns is true
    additional_ms2query_score_columns:
        These columns are appended to the standard columns and returned when return_non_classifier_columns is true
    """
    standard_columns = ["query_spectrum_nr", "ms2query_model_prediction", "precursor_mz_difference", "precursor_mz_query_spectrum",
                        "precursor_mz_analog", "inchikey", "spectrum_ids", "analog_compound_name"]
    if additional_metadata_columns is not None:
        standard_columns += additional_metadata_columns
    if additional_ms2query_score_columns is not None:
        standard_columns += additional_ms2query_score_columns
    classifier_columns = ["smiles", "cf_kingdom", "cf_superclass", "cf_class", "cf_subclass",
                          "cf_direct_parent", "npc_class_results", "npc_superclass_results", "npc_pathway_results"]
    if return_classifier_columns and return_non_classifier_columns:
        return standard_columns + classifier_columns
    if return_classifier_columns:
        return classifier_columns
    if return_non_classifier_columns:
        return standard_columns
    return []