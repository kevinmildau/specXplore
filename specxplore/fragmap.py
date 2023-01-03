# Developer Notes
# TODO: add max_spec = 50
#   Add a limiter to fragmap that prevents generation of a plot with more than 
#   50 spectra. At 50 spectra the y-axis becomes barely legible, and spectral 
#   differences crowd the plot so much that the x-axis requires very heavy zoom 
#   in, defeating the point of the visualization.
#   --> filter spectra to set of 50 most similar to root OR to 50 first indexed

from dash import html
from dash import dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matchms
# import pickle

def generate_fragmap_panel(spec_ids, all_spectra):
    if spec_ids and len(spec_ids) >= 2:
        # TODO: incorporate Henry's fragmap scripts.
        spectra = [all_spectra[i] for i in spec_ids]
        step = 0.1
        bins = [round(x, 1) for x in list(np.arange(0, 1000 + step, step))]
        fragmap = generate_fragmap(id_list=list(range(0, len(spectra))),
            spec_list=spectra, rel_intensity_threshold=0.00000,
            prevalence_threshold=0, mz_min=0, mz_max=1000,
            bins=bins)
        print(fragmap)
        out = [
            html.Div(dcc.Graph(id = "fragmap-panel", figure=fragmap, 
                style={"width":"100%","height":"60vh", 
                       "border":"1px grey solid"}))]
        return(out)
    else:
        out = [html.H6((
            "Select focus data and press" 
            "generate fragmap button for fragmap."))]
        return(out)

# generates long format data frame with spectral data columns id, m/z, 
# intensity
def spectrum_list_to_pandas(id_list: [int], 
    spec_list: [matchms.Spectrum]) -> pd.DataFrame:
    """
    Description
    """
    # Ensure the provided ID list is valid
    assert all([ind in range(0, len(spec_list)) for ind in id_list]), (
        f"Error: Provided ID list {id_list} contains ID's which do not" 
        f"match the spec_list of length {len(spec_list)}."
    )
    # Initialize empty list of data frames
    spec_data = list()
    for identifier in id_list:
        spec_data.append(pd.DataFrame({
            "spectrum": identifier, 
            "m/z": spec_list[identifier].mz,
            "intensity": spec_list[identifier].intensities}))
    # Return single long data frame
    long = pd.concat(objs=spec_data, ignore_index=True)
    return long



def bin_spectra(spectra: pd.DataFrame, bins: [float]) -> pd.DataFrame:
    """
    Description

    """

    # TODO: assert that column names are m/z, intensity, and ...

    # Bin Spectrum Data
    index_bin_map = pd.cut(
        x=spectra["m/z"], bins=bins, labels=bins[0:-1], include_lowest=True)
    spectra.insert(
        loc=2, column="bin", value=index_bin_map, allow_duplicates=True)

    # Return Binned Data as long pandas dataframe
    return spectra


def filter_binned_spectra_by_frequency(
    binned_spectra: pd.DataFrame, n_bin_cutoff: int):
    """
    Clean up binned data in place
    """
    # Count the total number of unique bins the dataset. Return if below threshold
    unique_bins = np.unique(binned_spectra['bin'].tolist())
    if len(unique_bins) <= n_bin_cutoff:
        return
    # Count the number of occurrences per bin
    bin_counts = binned_spectra.groupby(
        by="bin", axis=0, as_index=False, observed=True).size()
    bin_counts.sort_values(
        by="size", axis=0, ascending=False, inplace=True, ignore_index=True)

    # Filter Binned Spectra in place
    bins_to_keep = bin_counts.head(n=n_bin_cutoff)["bin"].tolist()
    binned_spectra.drop(
        labels=binned_spectra.loc[~binned_spectra["bin"].isin(bins_to_keep)].index, inplace=True)
    binned_spectra.reset_index(inplace=True, drop=True)


def get_precursors(
    id_list: [int], spec_list: [matchms.Spectrum]) -> {int: float}:
    # Initialize emtpy dictionary of precursors
    precursors = {spectrum: -1.0 for spectrum in id_list}
    # Iterate over all id's in ID-list
    for ind in id_list:
        # Identify the spectrum with the highest intensity, and save as precursor
        max_peak_index = np.argmax(spec_list[ind].intensities)
        precursors[ind] = spec_list[ind].mz[max_peak_index]
    # Return dictionary of precursors, indexed by spectrum ID's
    return precursors


def calculate_neutral_loss(
    spectrum: matchms.Spectrum, precursor: float) -> ([float], [float]):
    # Determine the number of peaks and the index of the precursor therein
    p_index, n_peaks = np.where(spectrum.mz == precursor), len(spectrum.mz)
    # Calculate the adjusted m/z values, but ignore the precursor index
    mz_adj = [
        abs(spectrum.mz[i] - precursor) 
        for i in range(0, n_peaks) 
        if i != p_index]
    intensities = [
        spectrum.intensities[i] 
        for i in range(0, n_peaks) 
        if i != p_index]
    # Return adjusted m/z values and corresponding intensities
    return mz_adj, intensities


def get_neutral_losses(id_list: [int], spec_list: [matchms.Spectrum]) -> pd.DataFrame:
    """
    Description
    """
    # Get Precursors
    precursors = get_precursors(id_list=id_list, spec_list=spec_list)
    # Initialize list of pandas to contain the neutral loss data
    neutral_loss_data = list()
    # Iterate over all indices specified or all indices in the data
    for ind in id_list:
        # Extra data, create pandas data frame, and append to growing list data frames
        neutral_loss_mz, neutral_loss_intensities = calculate_neutral_loss(
            spectrum=spec_list[ind], precursor=precursors[ind])
        neutral_loss_data.append(pd.DataFrame({
            "spectrum": ind,
            "m/z": neutral_loss_mz,
            "intensity": neutral_loss_intensities}))

    # Concatenate all data into singular long pandas
    return pd.concat(objs=neutral_loss_data, ignore_index=True)


def filter_binned_spectra(spectra: pd.DataFrame,
                          intensity_threshold: float,
                          prevalence_threshold: int,
                          mz_bounds: (float, float),
                          to_discard=None) -> [int]:
    """
    Description
    """
    # Initialize a list of bins to remove if they do not meet the criteria set
    to_discard = list() if to_discard is None else to_discard
    # Iterate over all bins in the current spectra dataset
    all_bins = np.unique(spectra["bin"].tolist())
    for current_bin in all_bins:
        # If the bin has already been flagged for removal, continue
        if current_bin in to_discard:
            continue
        # If the current bin's m/z values do not fall within the specified range, remove
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


def get_bin_map(bins: [int]):
    # Get all unique bins and sort them in ascending order
    unique_bins = np.unique(bins)
    unique_bins.sort()
    # Create mapping of bins to integers based on ascending order
    integer_map = {bin_id: index for index, bin_id in enumerate(unique_bins)}
    # Return a list of bins mapped to corresponding integer values
    return integer_map

def get_spectrum_map(spectra: [int], root: int):
    assert root in spectra, \
        f"Error: Root Spectrum {root} not present in provided spectrum list."
    # Get all unique spectra and sort them (without root, which will be the first)
    # TODO: other types of sorting in case spectrum ID's are not integers?
    unique_spectra = np.unique(spectra)
    unique_spectra = [
        spectrum for spectrum in unique_spectra if spectrum != root]
    unique_spectra.sort()
    # Create
    integer_map = {
        spectrum: index + 1 
        for index, spectrum in enumerate(unique_spectra) 
        if spectrum != root}
    integer_map[root] = 0
    return integer_map


def get_spectrum_plot_data(
    binned_spectra: pd.DataFrame, spectrum_map: {int: int}, 
    bin_map: {int: int}):
    """
    Description
    """
    # Map Bins and Spectra to their new continuous values for plotting
    new_data_frame = pd.DataFrame({
        "spectrum": [spectrum_map[spectrum] 
            for spectrum in binned_spectra["spectrum"].tolist()],
        "m/z": binned_spectra["m/z"].tolist(),
        "bin": [bin_map[bin_id] for bin_id in binned_spectra["bin"].tolist()],
        "intensity": binned_spectra["intensity"].tolist()
    })
    # Return Mapped Dataset
    return new_data_frame


def get_binned_spectrum_trace(binned_spectra: pd.DataFrame):
    # Create Heatmap Trace
    # TODO: parameterize the gaps in terms of the x and y axis instead of pixels
    heatmap_trace = go.Heatmap(x=binned_spectra["bin"].tolist(),
                               y=binned_spectra["spectrum"].tolist(),
                               z=binned_spectra["intensity"].tolist(),
                               xgap=5,
                               ygap=5, 
                               colorscale = "blugrn")

    # Return Heatmap Trace
    return heatmap_trace


def get_binned_neutral_trace(binned_spectra: pd.DataFrame):

    # Radius of Points
    r = 0.2

    # Create Point Scatter of neutral losses
    # 'line_color':'orange', 'line_width' : 0.5 # --> border lines with
    # different colors help visual clarity, but introduce border sizing
    # artefacts
    kwargs = {'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': 'red', 
        'opacity' : 1}  # TODO: ask kev why we using kwargs
    points = [
        go.layout.Shape(
            x0=binned_spectra.iloc[row]["bin"] - r,
            y0=binned_spectra.iloc[row]["spectrum"] - r,
            x1=binned_spectra.iloc[row]["bin"] + r,
            y1=binned_spectra.iloc[row]["spectrum"] + r,
            **kwargs) 
        for row 
        in range(0, binned_spectra.shape[0])]

    # Return the points trace
    return points


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

    # Step 1: Construct Binned Spectra
    spectra_long = spectrum_list_to_pandas(
        id_list=id_list, spec_list=spec_list)
    binned_spectra = bin_spectra(spectra=spectra_long, bins=bins)

    # Step 2: Filter Binned Spectra
    filter_binned_spectra_by_frequency(
        binned_spectra=binned_spectra, n_bin_cutoff=n_bin_cutoff)
    remaining_bins = np.unique(binned_spectra["bin"].tolist())
    print(binned_spectra)

    # Step 3-1: Calculate and Bin Neutral Losses
    neutral_losses = get_neutral_losses(id_list=id_list, spec_list=spec_list)
    binned_neutral = bin_spectra(spectra=neutral_losses, bins=bins)

    # Step 3-2: Keep only those bins featured in the binned spectrum set
    binned_neutral.drop(
        labels=binned_neutral.loc[
            ~binned_neutral["bin"].isin(remaining_bins)].index, 
            inplace=True)
    binned_neutral.reset_index(inplace=True, drop=True)

    # Step 4-1: Filter Binned Spectra
    discarded_bins = filter_binned_spectra(
        spectra=binned_spectra, intensity_threshold=rel_intensity_threshold,
        prevalence_threshold=prevalence_threshold,
        mz_bounds=(mz_min, mz_max))

    # Step 4-2: Filter Neutral Losses (propagate forward the already removed 
    # bins)
    # TODO: discuss with kev whether this forward propagation is necessary or 
    # desirable
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
    print(f"neutral mapped:")
    print(neutral_mapped)
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
        )
    )
    return fragmentation_map