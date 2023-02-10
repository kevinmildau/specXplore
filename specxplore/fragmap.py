from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matchms
from specxplore.specxplore_data import Spectrum
from typing import List, Union
import copy
from specxplore.compose import compose_function
from functools import partial
from warnings import warn
# import pickle

# TODO: REFACTOR THE PANEL GENERATOR
def generate_fragmap_panel(spectrum_identifier_list : List[int], all_spectra_list : List[Spectrum]) -> html.Div:
    """ Generates fragmap panel.
    """

    # 3 conditions
    # 1 --> empty or too small input; return empty
    # 2 --> input leads to fragmap build, return fragmap
    # 3 --> input is valid, but filtering prevents fragmap built, return empty
    if not spectrum_identifier_list or len(spectrum_identifier_list) < 2:
        empty_output_panel = [html.H6(("Select focus data and press generate fragmap button for fragmap."))]
        return empty_output_panel
    # Extract Spectra fro
    selected_spectra = [all_spectra_list[i] for i in spectrum_identifier_list]
    # Set bin settings TODO: do this somewhere more appropriate
    mass_to_charge_ratio_step_size = 0.1
    spectrum_bin_template = [
        round(x, 1) 
        for x in list(np.arange(0, 1000 + mass_to_charge_ratio_step_size, mass_to_charge_ratio_step_size))]
    # Call fragmap generator # TODO: add control interface for settings to app.py and add settings as input.
    # TODO: check whether passing actual identifier list works with downstream code indexing; it is unclear whether id_list
    # is actually a list of spectrum identifiers from all_spectra_list or whether it represents new iloc for the subset.
    fragmap = generate_fragmap(
        id_list=list(range(0, len(selected_spectra))), spec_list=selected_spectra, rel_intensity_threshold=0.00000,
        prevalence_threshold=0, mz_min=0, mz_max=1000, bins=spectrum_bin_template)
    fragmap_output_panel = [html.Div(dcc.Graph(id = "fragmap-panel", figure=fragmap, style={
        "width":"100%","height":"60vh", "border":"1px grey solid"}))]
    return(fragmap_output_panel)

# generates long format data frame with spectral data columns id, mz, intensity
def spectrum_list_to_pandas(spectrum_list: List[Spectrum]) -> pd.DataFrame:
    """ Constructs long format pandas data frame from spectrum list.

    :param spectrum_list: A list of Spectrum objects. The spectrum need to be binned already for aggregate lists to be
        available.
    :raises: Assertion error if is_binned_spectrum is False for a provided spectrum.
    :returns: Long format pandas data frame with all data needed for plotting fragmap.
    """

    # Check that all information required for data frame creation is present in spectrum object
    for spectrum in spectrum_list:
        assert spectrum.is_binned_spectrum == True, "spectrum list to pandas expects binned spectra as input."
    
    # Initialize empty list of data frames
    spectrum_dataframe_list = list()
    # Add spectra to data frame list
    for spectrum in spectrum_list:
        n_repeats = len(spectrum.mass_to_charge_ratios)
        spectrum_identifier_repeated = np.repeat(spectrum.identifier, n_repeats)
        is_binned_spectrum_repeated = np.repeat(spectrum.is_binned_spectrum, n_repeats)
        is_neutral_loss_repeated = np.repeat(spectrum.is_neutral_loss, n_repeats)
        tmp_df = pd.DataFrame({
            "spectrum_identifier": spectrum_identifier_repeated, 
            "mass_to_charge_ratio":spectrum.mass_to_charge_ratios, 
            "intensity":spectrum.intensities,
            "mass_to_charge_ratio_aggregate_list":spectrum.mass_to_charge_ratio_aggregate_list,
            "intensity_aggregate_list":spectrum.intensity_aggregate_list,
            "is_neutral_loss":is_neutral_loss_repeated,
            "is_binned_spectrum":is_binned_spectrum_repeated})
        spectrum_dataframe_list.append(tmp_df)
    # Concatenate data frame list
    long_pandas_df = pd.concat(objs=spectrum_dataframe_list, ignore_index=True)
    return long_pandas_df

# Pure Function.
def bin_spectrum(spectrum : Spectrum, bin_map : np.ndarray) -> Spectrum:
    """ Applies binning to mass_to_charge_ratio (mz) and intensity values and preserves aggregation information. 

    :param Spectrum: A spectrum object with mass_to_charge_ratios confined to the range of the bin_map.
    :param bin_map: An array with bins for mass to charge ratios, usually between 0 and 1000 with step_size of 0.1,
    :returns: A new spectrum object with binned spectra and aggregated data.

    Details:
    Constructs bin assignments for each mz value in the spectrum using bin_map. Then proceeds to loop through each
    unique bin assignment (i.e. index to bin_map). For each unique index, it gather all instances' data into 4 data
    containers:
        mass_to_charge_ratio_list --> list of unique binned mz values (multiple mz may be put into one bin)
        intensity_list --> list of additive intensities for each mz bin, renormalized as a final step
        mass_to_charge_ratio_aggregate_list -> List with sub lists of all mz values joined into single bin.
        intensity_aggregate_list -> List with sub lists of all intensity values joined into single bin.
    """

    mz_values = copy.deepcopy(spectrum.mass_to_charge_ratios)
    assert (max(mz_values) <= max(bin_map)) and (min(mz_values) >= min(bin_map)), ("All mz values in spectrum must be" 
        f" within range of bin_map, i.e. {min(bin_map)} and {max(bin_map)}")
    intensities = copy.deepcopy(spectrum.intensities)
    mz_value_bin_assignments = np.digitize(mz_values, bin_map, right=True)
    # Pre-allocation
    unique_assignments = np.unique(mz_value_bin_assignments)
    number_of_unique_assignments = len(unique_assignments)
    mass_to_charge_ratio_aggregate_list = [None] * number_of_unique_assignments
    intensity_aggregate_list = [None] * number_of_unique_assignments
    intensity_list = [0.0 for _ in range(0, number_of_unique_assignments)]
    mass_to_charge_ratio_list = [None for _ in range(0, number_of_unique_assignments)]
    # Double loop container assignment
    for idx_unique_assignments in range(0, number_of_unique_assignments):
        unique_assignment = unique_assignments[idx_unique_assignments] # this is an index for the bin_map
        for idx_bin_number in range(0, len(mz_value_bin_assignments)): # for all mz values / mz_bin_assignments
            # check whether the unique_assignment is a match to the assignment
            assignment = mz_value_bin_assignments[idx_bin_number]
            if unique_assignment == assignment:
                tmp_mz_bin = bin_map[assignment]
                tmp_mass_to_charge_ratio = mz_values[idx_bin_number]
                tmp_intensity = intensities[idx_bin_number]
                if mass_to_charge_ratio_aggregate_list[idx_unique_assignments] is None:
                    mass_to_charge_ratio_aggregate_list[idx_unique_assignments] = [tmp_mass_to_charge_ratio]
                else: 
                    mass_to_charge_ratio_aggregate_list[idx_unique_assignments].append(tmp_mass_to_charge_ratio)
                if intensity_aggregate_list[idx_unique_assignments] is None:
                    intensity_aggregate_list[idx_unique_assignments] = [tmp_intensity]
                else:
                    intensity_aggregate_list[idx_unique_assignments].append(tmp_intensity)
                intensity_list[idx_unique_assignments] += tmp_intensity
                if mass_to_charge_ratio_list[idx_unique_assignments] is None:
                    mass_to_charge_ratio_list[idx_unique_assignments] = tmp_mz_bin
    # Renormalization of intensities
    intensity_list = intensity_list / max(intensity_list)
    output_spectrum = Spectrum(
        mass_to_charge_ratios=np.array(mass_to_charge_ratio_list),
        precursor_mass_to_charge_ratio=copy.deepcopy(spectrum.precursor_mass_to_charge_ratio),
        identifier=copy.deepcopy(spectrum.identifier),
        intensities=np.array(intensity_list),
        intensity_aggregate_list=intensity_aggregate_list,
        mass_to_charge_ratio_aggregate_list=mass_to_charge_ratio_aggregate_list,
        is_neutral_loss = copy.deepcopy(spectrum.is_neutral_loss))
    return output_spectrum

# Pure function.
def compute_neutral_loss_spectrum(spectrum: Spectrum) -> Spectrum:
    """ Computes neutral loss mass to charge ratio values and corresponding intensities (nan)

    :param spectrum: A Spectrum object.
    :returns: A Spectrum object.
    """

    neutral_losses_mass_to_charge_ratios = abs(spectrum.precursor_mass_to_charge_ratio - spectrum.mass_to_charge_ratios)
    neutral_losses_mass_to_charge_ratios = np.array(
        [elem for elem in neutral_losses_mass_to_charge_ratios if elem != 0])
    neutral_losses_intensities = np.repeat(np.nan, neutral_losses_mass_to_charge_ratios.size)
    neutral_loss_spectrum = Spectrum(
        mass_to_charge_ratios = neutral_losses_mass_to_charge_ratios, 
        precursor_mass_to_charge_ratio = copy.deepcopy(spectrum.precursor_mass_to_charge_ratio), 
        identifier = copy.deepcopy(spectrum.identifier),
        intensities = neutral_losses_intensities,
        is_neutral_loss = True)
    return neutral_loss_spectrum

def generate_prevalence_filtered_binned_spectrum_df(
    spectrum_df : pd.DataFrame, n_min_occurrences : int) -> Union[pd.DataFrame, None]:
    """ Generate copy of spectrum_df with row filtered such that each mz instance occurs at least n_min_occurrences 
    times. 
    """

    assert n_min_occurrences >=1 and isinstance(n_min_occurrences,int), (
        "n_min_occurrences must be an integer above or equal to 1.")
    if n_min_occurrences == 1: # A single occurence of a fragment is enough for inclusion, return input.
        return spectrum_df
    if spectrum_df is None or spectrum_df.empty: # no data provided, return None
        return None
    if n_min_occurrences > np.unique(spectrum_df["spectrum_identifier"]).size:
        warn("n_min_occurrences across spectra for fragment exceeds number of spectra. Return None.", UserWarning)
        return None
    output_spectrum_df = copy.deepcopy(spectrum_df)
    output_spectrum_df = output_spectrum_df.groupby('mass_to_charge_ratio').filter(lambda x: len(x) >= n_min_occurrences)
    if output_spectrum_df.empty: # no data left after filtering, return None
        return None
    output_spectrum_df.reset_index(inplace=True, drop=True)
    return output_spectrum_df

def generate_mz_range_filtered_binned_spectrum_df(spectrum_df : pd.DataFrame, mz_min : float, mz_max : float):
    """ Generate copy of spectrum_df with rows filtered to have mz values in specified range."""

    assert mz_min < mz_max, "mz_min must be strictly smaller than mz_max"
    if spectrum_df is None or spectrum_df.empty: # no data provided, return None
        return None
    output_spectrum_df = copy.deepcopy(spectrum_df)
    selection_map = [x >= mz_min and x <= mz_max for x in output_spectrum_df["mass_to_charge_ratio"]]
    output_spectrum_df = output_spectrum_df.loc[selection_map]
    if output_spectrum_df.empty: # no data left after filtering, return None
        return None
    output_spectrum_df.reset_index(inplace=True, drop=True)
    return output_spectrum_df

def generate_intensity_filtered_binned_spectrum_df(spectrum_df : pd.DataFrame, intensity_min : float):
    """ Generate copy of spectrum_df with fragments filtered by minimum intensity. """

    if spectrum_df is None or spectrum_df.empty: # no data provided, return None
        return None
    output_spectrum_df = copy.deepcopy(spectrum_df)
    selection_map = [x >= intensity_min for x in output_spectrum_df["intensity"]]
    output_spectrum_df = output_spectrum_df.loc[selection_map]
    if output_spectrum_df.empty: # no data left after filtering, return None
        return None
    output_spectrum_df.reset_index(inplace=True, drop=True)
    return output_spectrum_df

# TODO: INCORPORATE INTO GENERATE HEATMAP FUNCTION
# generate_x_axis_labels_for_bins()
# bins = all_observed_mz_values (binned)
def get_bin_map(bins: [int]):
    # Get all unique bins and sort them in ascending order
    unique_bins = np.unique(bins)
    unique_bins.sort()
    # Create mapping of bins to integers based on ascending order
    integer_map = {bin_id: index for index, bin_id in enumerate(unique_bins)}
    # Return a list of bins mapped to corresponding integer values
    return integer_map

# TODO: INCORPORATE INTO GENERATE HEATMAP FUNCTION
def get_heatmap(spectra_df: pd.DataFrame):
    """ Generate Heatmap Trace """
    
    tmp_df = copy.deepcopy(spectra_df)
    # construct dense ordering of mass_to_charge ratios for heatmap x values
    tmp_df["bin_index"] = np.array(tmp_df["mass_to_charge_ratio"].rank(method="dense"), dtype=np.int64)
    tmp_df["spec_index"] = np.array(tmp_df["spectrum_identifier"].rank(method="dense"), dtype=np.int64)
    

    def list_to_string(input_list : List):
        return ' & '.join(map(str, input_list))
    


    text = [(f"Original m/z values: {list_to_string(np.round(x,3))},"
        f" <br>Original intensity values: {list_to_string(np.round(y,3))}")
        for x,y,z in zip(
            tmp_df["mass_to_charge_ratio_aggregate_list"].tolist(), 
            tmp_df["intensity_aggregate_list"].tolist(),
            tmp_df["is_neutral_loss"].tolist()) if not z]
    
    
    heatmap_trace = go.Heatmap(x=tmp_df["bin_index"].to_list(),
                               y=tmp_df["spec_index"].to_list(),
                               z=tmp_df["intensity"].to_list(),
                               text = text, xgap=5, ygap=5, 
                               colorscale = "blugrn")

    fragmap_figure = go.Figure(data=[heatmap_trace])
    
    radius = 0.1
    # TODO: ADD N_ROWS PARAMETER

    # Create Point Scatter of neutral losses
    # 'line_color':'orange', 'line_width' : 0.5 # --> border lines with
    # different colors help visual clarity, but introduce border sizing
    # artefacts

    
    tmp_df_neutral_losses = tmp_df.loc[tmp_df["is_neutral_loss"]]

    print("CHECKPOINT")
    kwargs = {'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': 'red', 'opacity' : 1}
    shapes = [ go.layout.Shape( x0=tmp_df_neutral_losses.iloc[row]["bin_index"] - radius,
            y0=tmp_df_neutral_losses.iloc[row]["spec_index"] - radius,
            x1=tmp_df_neutral_losses.iloc[row]["bin_index"] + radius,
            y1=tmp_df_neutral_losses.iloc[row]["spec_index"] + radius,
            **kwargs) 
        for row in range(0, tmp_df_neutral_losses.shape[0])]
    #print(shapes)

    fragmap_figure.update_layout(
        # Add Neutral Loss Points Trace
        shapes=shapes,
        # Change Theme
        template="simple_white", 
        # Lock aspect ratio of heatmap and relabel the y-axis
        yaxis=dict(
            #scaleanchor='x',
            fixedrange=True,
            tickmode='array',
            tickvals=np.unique(tmp_df["spec_index"]),
            ticktext=np.unique(tmp_df["spectrum_identifier"])
        ),
        # relabel the x-axis
        xaxis=dict(
            tickmode='array',
            tickvals=np.unique(tmp_df["bin_index"]),
            ticktext=np.unique(tmp_df["mass_to_charge_ratio"]),
            #tickvals=list(bin_map.values()),
            #ticktext=list(bin_map.keys())
        ),
        margin = {"autoexpand":True, "b" : 10, "l":10, "r":10, "t":10}
    )
    # Return Heatmap Trace
    return fragmap_figure


# TODO: INCORPORATE INTO GENERATE HEATMAP FUNCTION
# generate_neutral_loss_shape_list()
# THIS IS NOT THE BINNED SPECTRA, BUT BINNED NEUTRAL LOSSES NOW CONCATENATED TO A PD.DATAFRAME
# THE NEUTRAL NEED TO BE BINNED
def get_binned_neutral_trace(binned_spectra: pd.DataFrame):

    # Radius of Points
    r = 0.2
    # TODO: ADD N_ROWS PARAMETER

    # Create Point Scatter of neutral losses
    # 'line_color':'orange', 'line_width' : 0.5 # --> border lines with
    # different colors help visual clarity, but introduce border sizing
    # artefacts
    kwargs = {'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': 'red', 'opacity' : 1}
    shapes = [ go.layout.Shape( x0=binned_spectra.iloc[row]["bin"] - r,
            y0=binned_spectra.iloc[row]["spectrum"] - r,
            x1=binned_spectra.iloc[row]["bin"] + r,
            y1=binned_spectra.iloc[row]["spectrum"] + r,
            **kwargs) 
        for row in range(0, binned_spectra.shape[0])]

    # Return the points trace
    return shapes

def generate_fragmap(
    spectrum_list : List[Spectrum], relative_intensity_threshold : float, prevalence_threshold : int,
    mass_to_charge_ratio_minimum : float, mass_to_charge_ratio_maximum : float, bin_map : List[float],
    root_spectrum_identifier : int, max_number_of_binned_fragments : int = 200) -> None:
    """ Processes list of spectra and generates fragmap graph. 

    :param spectrum_list: List of Spectrum objects with mass_to_charge_ratios and intensities arrays.
    :param relative_intensity_threshold: Minimum relative peak intensity after binning for binned fragment to be 
        included in plot.
    :param prevalence_threshold: Minimum occurence number accross spectra for binned fragment to be included in plot.
    :param mass_to_charge_ratio_minimum: Minimum mass to charge ratio considered for plotting. All lower values will be 
        cut.
    :param mass_to_charge_ratio_maximum: Maximum mass to charge ratio considered for plotting. All higher values will be
        cut.
    :param bin_map: List with sequence of mass to charge ratio values to be used for binning. Usually a sequence from
        0 to 1000 in steps of 0.1, e.g. 0, 0.1, 0.2, 0.3 ... 999.8, 999.9, 1000.
    :param root_spectrum_identifier: Spectrum identifier for spectrum to be used as lowest entry in fragmap.
    :param max_number_of_binned_fragments: Maximum number of binned fragments per Spectrum considered. If, after all
        other filtering steps, a spectrum exceeds this number of fragments, the spectrum is truncated to the fragments
        with the largest 200 intensities (i.e. low intensity noise fragment removal). Deactivating this behavior can be 
        achieved by setting the max number of fragments very high (e.g. 9999), but may lead to unreadable fragmaps.
    """
    # TODO: CONSIDER: prefilter spectra to size 200 before doing all data processing. Requires additional input?

    # Get binned spectra and binned neutral loss spectra
    binned_spectrum_list = [bin_spectrum(spectrum, bin_map) for spectrum in spectrum_list]
    neutral_loss_spectrum_list = [
        bin_spectrum(compute_neutral_loss_spectrum(spectrum), bin_map) for spectrum in spectrum_list]
 
   
    # construct spectra df and neutral loss df
    spectra_df = spectrum_list_to_pandas(binned_spectrum_list)
    losses_df = spectrum_list_to_pandas(neutral_loss_spectrum_list)

    # Compose filter pipeline function using provided settings
    filter_pipeline_spectra = compose_function(
        partial(generate_intensity_filtered_binned_spectrum_df, intensity_min=relative_intensity_threshold),
        partial(generate_mz_range_filtered_binned_spectrum_df, 
            mz_min = mass_to_charge_ratio_minimum, mz_max = mass_to_charge_ratio_maximum))
    
    filtered_spectra_df = filter_pipeline_spectra(spectra_df)
    filtered_losses_df = generate_mz_range_filtered_binned_spectrum_df(
        losses_df, mz_min = mass_to_charge_ratio_minimum, mz_max = mass_to_charge_ratio_maximum)

    all_plot_data_df = pd.concat([filtered_spectra_df, filtered_losses_df], axis = 0)
    all_plot_data_df = generate_prevalence_filtered_binned_spectrum_df(all_plot_data_df, n_min_occurrences=1)
    
    fragmap = get_heatmap(all_plot_data_df)


    return fragmap
    
    # construct plotly graph
    # return plotly graph
