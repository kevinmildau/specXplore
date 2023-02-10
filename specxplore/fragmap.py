from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matchms
from specxplore.specxplore_data import Spectrum
import typing
from typing import List, Tuple
import copy
# import pickle

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
    """
    Constructs long format pandas data frame from spectrum list.
    """
    # Initialize empty list of data frames
    spectrum_dataframe_list = list()
    for spectrum in spectrum_list:
        spectrum_dataframe_list.append(pd.DataFrame({
            "spectrum_identifier": spectrum.identifier, 
            "mass-to-charge-ratio": spectrum.mass_to_charge_ratios, 
            "intensity": spectrum.intensities }))
    # Return single long data frame
    long_pandas_df = pd.concat(objs=spectrum_dataframe_list, ignore_index=True)
    return long_pandas_df

# Pure Function.
def bin_spectrum(spectrum : Spectrum, bin_map : np.ndarray) -> Spectrum:
    """ Applies binning to mass-to-charge-ratio (mz) and intensity values and preserves aggregation information. 

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
        mass_to_charge_ratio_aggregate_list=mass_to_charge_ratio_aggregate_list)
    return output_spectrum

# PURE FUNCTION
def bin_spectra(spectra_data_frame: pd.DataFrame, bins: List[float]) -> pd.DataFrame:
    """
    Input is a dataframe with multiple a single or multiple spectra. Bin values are 
    """
    # GENERALIZE TO WORK WITH SINGLE SPECTRUM OBJECT
    # INPUT IS A SPECTRUM OBJECT, OUTPUT IS A SPECTRUM OBJECT. 
    # GENERALIZE AT A LATER STAGE TO: OUTPUT IS A BINNED SPECTRUM OBJECT WITH TUPLES OF BINNED VALUES ACCESSIBLE

    # Bin Spectrum Data

    spectra_data_frame_copy = copy.deepcopy(spectra_data_frame)
    
    index_bin_map = pd.cut(x=spectra_data_frame_copy["mass-to-charge-ratio"], bins=bins, labels=bins[0:-1], include_lowest=True)
    
    spectra_data_frame_copy.insert(loc=2, column="bin", value=index_bin_map, allow_duplicates=True)

    # Return Binned Data as long pandas dataframe
    return spectra_data_frame_copy

def filter_binned_spectra_by_frequency(binned_spectra: pd.DataFrame, n_bin_cutoff: int):
    """
    Clean up binned data in place
    """
    # TODO: EVALUATE BEST PLACE FOR THIS FILTER. 
    # IF LIST OF SPECTRA IS INPUT:
    #   1) construct set of all unique possible fragments (mz) - loop over list of mz vals arrays
    #   --> as you do, keep track of occurence number
    #   2) loop through occurence number structure and extract only bins above threshold (set)
    #   3) loop through all spectra once more and create replacement mz bins limited by occurence set

    # first find out which mz values (and correspondingly intensity values) should be kept
    # then filter the mz values of each spectrum down to this selection
    # finally check whether there are now empty spectra, if so, remove; 
    # if there are no spectra left at all, pass toward empty div

    # TODO: turn input into list of spectra (binned, but still simple spectra)
    # Simply a filter for max number of mz int tuples by sorted intensity

    # Count the total number of unique bins the dataset. Return if below threshold
    unique_bins = np.unique(binned_spectra['bin'].tolist())
    if len(unique_bins) <= n_bin_cutoff:
        return # <-- # TODO: FIX NO OUTPUT DEPENDING ON INPUT TO MORE PREDICTABLE RETURN. return binned_spectra ?
    
    # Count the number of occurrences per bin
    bin_counts = binned_spectra.groupby(by="bin", axis=0, as_index=False, observed=True).size()
    bin_counts.sort_values(by="size", axis=0, ascending=False, inplace=True, ignore_index=True)

    # Filter Binned Spectra in place
    bins_to_keep = bin_counts.head(n=n_bin_cutoff)["bin"].tolist()
    binned_spectra.drop(
        labels=binned_spectra.loc[~binned_spectra["bin"].isin(bins_to_keep)].index, inplace=True)
    binned_spectra.reset_index(inplace=True, drop=True)

# specxplore dataclass spectrum
# generate only neutral losses --> list of mz, list of intensities ==> np.arrays()
def compute_neutral_loss_spectrum(spectrum: Spectrum) -> Spectrum:
    """ Computes neutral loss mass to charge ratio values and corresponding intensities (nan)
    """
    neutral_losses_mass_to_charge_ratios = abs(spectrum.precursor_mass_to_charge_ratio - spectrum.mass_to_charge_ratios)
    neutral_losses_intensities = np.repeat(np.nan, neutral_losses_mass_to_charge_ratios.size)

    neutral_loss_spectrum = Spectrum(
        mass_to_charge_ratios = neutral_losses_mass_to_charge_ratios, 
        precursor_mass_to_charge_ratio = copy.deepcopy(spectrum.precursor_mass_to_charge_ratio), 
        identifier = copy.deepcopy(spectrum.identifier),
        intensities = neutral_losses_intensities,
        is_neutral_loss = True)
    return neutral_loss_spectrum

# get as input precursor and mz and int; do not require subselection in this; only pass relevant
# generate list of datclass spectrums
# create func to generate the pandas df / whatever needed for the plot
def get_neutral_losses_data_frame(spec_list: List[Spectrum]) -> pd.DataFrame:
    """
    Description
    """
    # Initialize list of pandas data frames to contain the neutral loss data
    neutral_losses_pandas_list = list()

    # Iterate over all indices specified or all indices in the data
    for spectrum in spec_list:
        # Extra data, create pandas data frame, and append to growing list data frames
        neutral_losses_mass_to_charge_ratios, neutral_losses_intensities = compute_neutral_losses(spectrum=spectrum)
        neutral_losses_pandas_list.append(pd.DataFrame({
            "spectrum": spectrum.identifier, # TODO: DOUBLE CHECK FOR ILOC IN LOCAL SET VS GLOBAL SET CONSISTENCY
            "mass-to-charge-ratios": neutral_losses_mass_to_charge_ratios,
            "intensity": neutral_losses_intensities}))
    # Concatenate all data into singular long pandas
    long_pandas = pd.concat(objs=neutral_losses_pandas_list, ignore_index=True)
    return long_pandas

# THIS CAN BE REPLACED WITH MATCHMS FILTERS FOR MZ BOUNDS, INTENSITY AND MAX NUMBER OF PEAKS
# 3 lines of code.
def filter_binned_spectra(spectra: pd.DataFrame,
                          intensity_threshold: float,
                          prevalence_threshold: int,
                          mz_bounds: (float, float),
                          to_discard=None) -> [int]:
    """
    Description
    """
    #for each spec:
    #    filter_intensity (each spectrum) # get ridd of noise
    #    filter_mz (each spectrum) # limit to range
    
    # filter prevalence would work in a single line when using pd filters
    #filter_prevalence (list of spectra) # avoid non-overlaps

    # Initialize a list of bins to remove if they do not meet the criteria set
    to_discard = list() if to_discard is None else to_discard
    # Iterate over all bins in the current spectra dataset
    all_bins = np.unique(spectra["bin"].tolist())
    for current_bin in all_bins:
        # If the bin has already been flagged for removal, continue
        if current_bin in to_discard:
            continue
        # If the current bin's mz values do not fall within the specified range, remove
        if mz_bounds[0] > current_bin or mz_bounds[1] <= current_bin:
            to_discard.append(current_bin)
            continue
        # Extract all data of the current bin
        bin_data = spectra.loc[spectra['bin'] == current_bin]
        # If the bin has too few spectra featured within it, discard
        if bin_data.shape[0] < prevalence_threshold:
            to_discard.append(current_bin)
            continue
        # If all the bin's intensities are below the threshold, discard
        if all(intensity < intensity_threshold for intensity in bin_data["intensity"].tolist()):
            to_discard.append(current_bin)
            continue
    # Filter out all the
    spectra.drop(
        labels=spectra.loc[spectra["bin"].isin(to_discard)].index, 
        inplace=True)
    spectra.reset_index(inplace=True, drop=True)
    # Return the list of bins discarded
    return to_discard


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

# generate_y_axis_labels_for_specs
# input to get_spectrum_map should ??? contain actual spec_ids for y axis labelling.
# beware of non-iloc downstream effects
def get_spectrum_map(spectra: [int], root: int):
    assert root in spectra, (
        f"Error: Root Spectrum {root} not present in provided spectrum list."
    )
        
    # Get all unique spectra and sort them (without root, which will be the first)
    # TODO: other types of sorting in case spectrum ID's are not integers?
    
    # pandas specific artefacts since ids may now be duplicated
    unique_spectra = np.unique(spectra)
    unique_spectra = [ spectrum for spectrum in unique_spectra if spectrum != root]

    unique_spectra.sort()
    
    # TODO: double check indexing validity
    integer_map = {
        spectrum: index + 1 
        for index, spectrum in enumerate(unique_spectra) 
        if spectrum != root}
    
    integer_map[root] = 0
    return integer_map

# plot data generator function that creates input for heatmap
# this is where the pandas aggregation should happen
def get_spectrum_plot_data(
    binned_spectra: pd.DataFrame, spectrum_map: {int: int}, 
    bin_map: {int: int}):
    """
    Description
    """
    # Map Bins and Spectra to their new continuous values for plotting


    # create list of mapped spectra
    # 1 row for each spec_id and y_index tuple
    # key indexing, get for each spectrum_identifier the corresponding y_index continuous position

    # [spectrum_map[spectrum] for spectrum in binned_spectra["spectrum"].tolist()]

    new_data_frame = pd.DataFrame({
        "spectrum": [spectrum_map[spectrum] for spectrum in binned_spectra["spectrum"].tolist()],
        "bin": [bin_map[bin_id] for bin_id in binned_spectra["bin"].tolist()], # integer for x_axis
        "mass-to-charge-ratio": binned_spectra["mass-to-charge-ratio"].tolist(), # --> actual mz values for each x_axis thing; could be tuple
        "intensity": binned_spectra["intensity"].tolist()
    })
    # Return Mapped Dataset
    return new_data_frame


def get_binned_spectrum_trace(binned_spectra: pd.DataFrame):
    # Create Heatmap Trace
    # TODO: parameterize the gaps in terms of the x and y axis instead of pixels
    # TODO: ADD HOVERETEMPLATE TO GIVE MEANING TO TEXT ADDITION (LIST OF MZ VALUES AGGREGATED INTO BIN)
    heatmap_trace = go.Heatmap(x=binned_spectra["bin"].tolist(),
                               y=binned_spectra["spectrum"].tolist(),
                               z=binned_spectra["intensity"].tolist(),
                               text = binned_spectra["mass-to-charge-ratio"].tolist(),
                               xgap=5,
                               ygap=5, 
                               colorscale = "blugrn")

    # Return Heatmap Trace
    return heatmap_trace


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
    kwargs = {'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': 'red', 
        'opacity' : 1}
    shapes = [
        go.layout.Shape(
            x0=binned_spectra.iloc[row]["bin"] - r,
            y0=binned_spectra.iloc[row]["spectrum"] - r,
            x1=binned_spectra.iloc[row]["bin"] + r,
            y1=binned_spectra.iloc[row]["spectrum"] + r,
            **kwargs) 
        for row 
        in range(0, binned_spectra.shape[0])]

    # Return the points trace
    return shapes


def generate_fragmap(id_list: [int], spec_list: [matchms.Spectrum], 
    rel_intensity_threshold: float, prevalence_threshold: int, mz_min: float,
    mz_max: float, bins: [float], root: int = 0, n_bin_cutoff: int = 200):
    """
    Wrapper function that executes complete generate fragmap routine and 
    returns final fragmap as plotly graph object.

    Arguments:
        id_list: A list of spectrum id's (int list index)
        spec_list: An ordered list of matchms spectra (aligned with numeric 
            list index)
        rel_intensity_threshold: Float input. Minimum value of intensity a 
            fragment needs to reach in at least one spectrum.
        prevalence_threshold: integer input. Minimum number of occurences of 
            bin across selected spectrum set.
        mz_max: Maximum mz value for a bin to be considered.
        mz_min: Minimum mz value for a bin to be considered.
        bins: a list of floats / integers mapping out the bin edges to be used
        root: 0 - Index of the primary spectrum. For now assumed 0.
        n_bin_cutoff: the maximum number of bins to display
    """


    # filter raw data - add settings to dashboard
    # filter prevalence - make optional, add setting to dashboard
    
    # construct set() of bins --> input for fast neutral loss suitability checks
    
    # neutral loss computation
    # neutral losses should be exclusively in the existing bin set


    # Step 1: Construct Binned Spectra
    spectra_long = spectrum_list_to_pandas(
        id_list=id_list, spec_list=spec_list)
    binned_spectra = bin_spectra(spectra=spectra_long, bins=bins)

    # Step 2: Filter Binned Spectra
    filter_binned_spectra_by_frequency(
        binned_spectra=binned_spectra, n_bin_cutoff=n_bin_cutoff)
    remaining_bins = np.unique(binned_spectra["bin"].tolist())


    # Step 3-1: Calculate and Bin Neutral Losses
    neutral_losses = get_neutral_losses_data_frame(spec_list=spec_list)
    binned_neutral = bin_spectra(spectra=neutral_losses, bins=bins)

    # Step 3-2: Keep only those bins featured in the binned spectrum set
    binned_neutral.drop(
        labels=binned_neutral.loc[
            ~binned_neutral["bin"].isin(remaining_bins)].index, 
            inplace=True)
    binned_neutral.reset_index(inplace=True, drop=True)


    # THIS SHOULD BE IN THE BEGINNING TO LIMIT NUMBER OF FRAGMENTS CONSIDERED
    # Step 4-1: Filter Binned Spectra
    discarded_bins = filter_binned_spectra(
        spectra=binned_spectra, intensity_threshold=rel_intensity_threshold,
        prevalence_threshold=prevalence_threshold,
        mz_bounds=(mz_min, mz_max))

    # THIS SECOND STEP SEEMS SUPERFLUOUS
    #
    # Step 4-2: Filter Neutral Losses (propagate forward the already removed 
    # bins)
    discarded_bins = filter_binned_spectra(
        spectra=binned_neutral, intensity_threshold=rel_intensity_threshold,
        prevalence_threshold=prevalence_threshold, mz_bounds=(mz_min, mz_max),
        to_discard=discarded_bins)

    
    # Step 5-1: Create Map of Bins and Spectra to Integers
    spectrum_map = get_spectrum_map(
        spectra=binned_spectra["spectrum"].tolist(), root=root)
    bin_map = get_bin_map(bins=binned_spectra["bin"].tolist())
    # Step 5-2: Map both binned spectra and neutral losses dataframes to the 
    # new integer set
    spectra_mapped = get_spectrum_plot_data(
        binned_spectra=binned_spectra, spectrum_map=spectrum_map, 
        bin_map=bin_map)
    neutral_mapped = get_spectrum_plot_data(
        binned_spectra=binned_neutral, spectrum_map=spectrum_map, 
        bin_map=bin_map)



    # construct_heatmap_df (contains both mz and neutral losses)
    # apply_prevalence_filter to heatmap_df
    # isolate trace data --> mz_df, neutral_df


    # plot objects (with separate dfs)
    # --> two trace generations
    # Add layout and styling to plot function
    
    # Step 6-1: Create Individual Traces
    heatmap_trace = get_binned_spectrum_trace(binned_spectra=spectra_mapped)
    points_trace = get_binned_neutral_trace(binned_spectra=neutral_mapped)
    # Step 6-2: Combine Traces and Style Figure
    fragmentation_map = go.Figure(data=[heatmap_trace])
    fragmentation_map.update_layout(
        # Add Neutral Loss Points Trace
        shapes=points_trace,
        # Change Theme
        template="simple_white", 
        # Lock aspect ratio of heatmap and relabel the y-axis
        yaxis=dict(
            #scaleanchor='x',
            fixedrange=True,
            tickmode='array',
            tickvals=list(spectrum_map.values()),
            ticktext=list(spectrum_map.keys())
        ),
        # relabel the x-axis
        xaxis=dict(
            tickmode='array',
            tickvals=list(bin_map.values()),
            ticktext=list(bin_map.keys())
        ),
        margin = {"autoexpand":True, "b" : 10, "l":10, "r":10, "t":10}
    )
    return fragmentation_map