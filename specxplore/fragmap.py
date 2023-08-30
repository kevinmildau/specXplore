from dash import html, dcc
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from specxplore.importing import Spectrum, SpectraDF, filter_spectrum_top_k_intensity_fragments
from typing import List, Union
import copy

# TODO: REFACTOR THE PANEL GENERATOR
def generate_fragmap_panel(
        spectrum_identifier_list : List[int], 
        all_spectra_list : List[Spectrum], 
        top_k : int
        ) -> html.Div:
    """ 
    Generate fragmap panel.

    Parameters:
        spectrum_identifier_list: List of integers giving iloc of the spectra to be used in fragmap.
        all_spectra_list: List of all spectra from which to extract the spectra selected using spectrum_identifier_list.
    Returns:
        A html.Div with a fragmap dcc.Graph inside or a html.Div with a data request text.
    Raises:
        None.
    """

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
    fragmap = generate_fragmap(selected_spectra, 0.0, 1, 0, 1000, spectrum_bin_template, top_k)
    fragmap_output_panel = [dcc.Graph(id = "fragmap-panel", figure=fragmap, style={
        "width":"100%","height":"100%", "border":"1px grey solid"})]
    return(fragmap_output_panel)

# generates long format data frame with spectral data columns id, mz, intensity
def spectrum_list_to_pandas(spectrum_list: List[Spectrum]) -> SpectraDF:
    """ 
    Constructs SpectraDF with long format pandas data frame from spectrum list.

    Parameters:
        spectrum_list: A list of Spectrum objects. The spectrum need to be binned already for aggregate lists to be
        available.
    Raises: 
        ValueError: if is_binned_spectrum is False for a provided spectrum.
    Returns: 
        SpectraDF object with long format pandas data frame with all data needed for plotting fragmap.
    """

    # Check that all information required for data frame creation is present in spectrum object
    for spectrum in spectrum_list:
        assert spectrum.is_binned_spectrum == True, "spectrum list to pandas expects binned spectra as input."
    # Initialize empty list of data frames
    spectrum_dataframe_list = list()
    # Add spectra to data frame list
    for spectrum in spectrum_list:
        n_repeats = len(spectrum.mass_to_charge_ratios)
        spectrum_identifier_repeated = np.repeat(spectrum.spectrum_iloc, n_repeats)
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
    output_container = SpectraDF(pd.concat(objs=spectrum_dataframe_list, ignore_index=True))
    output_container = output_container.astype({"spectrum_identifier": np.int64})
    return output_container

# Pure Function.
def bin_spectrum(spectrum : Spectrum, bin_map : np.ndarray) -> Spectrum:
    """ 
    Applies binning to mass_to_charge_ratio (mz) and intensity values and preserves aggregation information. 

    Parameters:
        Spectrum: Spectrum object with mass_to_charge_ratios confined to the range of the bin_map.
        bin_map: Array with bins for mass to charge ratios, usually between 0 and 1000 with step_size of 0.1,
    Returns: 
        Spectrum object with binned spectra and aggregated data.

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
        spectrum_iloc=copy.deepcopy(spectrum.spectrum_iloc),
        feature_id=copy.deepcopy(spectrum.feature_id),
        intensities=np.array(intensity_list),
        intensity_aggregate_list=intensity_aggregate_list,
        mass_to_charge_ratio_aggregate_list=mass_to_charge_ratio_aggregate_list,
        is_neutral_loss = copy.deepcopy(spectrum.is_neutral_loss))
    return output_spectrum

# Pure function.
def compute_neutral_loss_spectrum(spectrum: Spectrum) -> Spectrum:
    """ 
    Computes neutral loss mass to charge ratio values and corresponding intensities (nan)

    Parameters:
        spectrum: Spectrum object.
    Returns: 
        Spectrum object.
    """

    neutral_losses_mass_to_charge_ratios = abs(spectrum.precursor_mass_to_charge_ratio - spectrum.mass_to_charge_ratios)
    neutral_losses_mass_to_charge_ratios = np.array(
        [elem for elem in neutral_losses_mass_to_charge_ratios if elem != 0])
    neutral_losses_intensities = np.repeat(np.nan, neutral_losses_mass_to_charge_ratios.size)
    neutral_loss_spectrum = Spectrum(
        mass_to_charge_ratios = neutral_losses_mass_to_charge_ratios, 
        precursor_mass_to_charge_ratio = copy.deepcopy(spectrum.precursor_mass_to_charge_ratio), 
        spectrum_iloc = copy.deepcopy(spectrum.spectrum_iloc),
        feature_id= copy.deepcopy(spectrum.feature_id),
        intensities = neutral_losses_intensities,
        is_neutral_loss = True)
    return neutral_loss_spectrum

def generate_prevalence_filtered_df(
    spectra_df : Union[SpectraDF, None], n_min_occurrences : int) -> Union[SpectraDF, None]:
    """ 
    Generate copy of spectrum_df with row filtered such that each mz instance occurs at least n_min_occurrences times. 
    
    Parameters:
        spectrum_df: A SpectraDF object containing pandas.DataFrame.
        losses_df: A SpectraDF object containing pandas.DataFrame.
        n_min_occurences: Integer, minimum number of occurrences of the same mass_to_charge_ratio value for it to be
            kept in the dataframe in filtering step.

    Developer Note:
    Neutral losses and mass to charge ratios are considered equals in this function. That is, if the minumum number of
    occurrences is set to 2, both unique neutral losses and unique observed fragments will be removed from the data to
    be visualized. A re-introduction step for any observed fragments could be added via joining the is_loss = False
    row subset of the dataframe with the filtered set (inner join).
    """

    assert n_min_occurrences >=1 and isinstance(n_min_occurrences, int), (
        "n_min_occurrences must be an integer above or equal to 1.")
    if spectra_df == None: 
        return None
    output_df = spectra_df.get_data()
    #if n_min_occurrences > np.unique(output_df["spectrum_identifier"]).size:
    #    warn("n_min_occurrences exceeds number of spectra. Return None.", UserWarning)
    #    return None
    output_df = output_df.groupby('mass_to_charge_ratio').filter(lambda x: len(x) >= n_min_occurrences)
    # TODO: Consider adding optional rejoin of filtered out real spectra with is_loss == False subset inner join. 
    #       Beware of issues empty df issues with this approach however.
    if output_df.empty: # no data left after filtering, return None
        return None
    output_df.reset_index(inplace=True, drop=True)
    return SpectraDF(output_df)

def generate_mz_range_filtered_df(
        spectra_df : Union[SpectraDF, None], 
        mz_min : float, 
        mz_max : float
        ) -> Union[SpectraDF, None]:
    """ 
    Generate copy of spectrum_df with rows filtered to have mz values in specified range.
    
    Parameters:
        spectrum_df: A SpectraDF object with pandas.DataFrame
        mz_min: float, minimum value of mz. All values lower will be removed from df.
        mz_max: flaot, maximum value of mz. All values above will be removed from df.
    Returns:
        New SpectraDF object with filtered pandas data frame. 
        Returns None if data frame input is None or output is empty.
    Raises:
        ValueError: if mz_min > mz_max.    
    """

    assert mz_min < mz_max, "mz_min must be strictly smaller than mz_max"
    output_df = spectra_df.get_data()

    if output_df is None or output_df.empty: # no data provided, return None
        return None
    selection_map = [x >= mz_min and x <= mz_max for x in output_df["mass_to_charge_ratio"]]
    output_df = output_df.loc[selection_map]
    if output_df.empty: # no data left after filtering, return None
        return None
    output_df.reset_index(inplace=True, drop=True)
    return SpectraDF(output_df)

def generate_intensity_filtered_df(
        spectra_df : Union[SpectraDF, None], 
        intensity_min : float
        ) -> Union[SpectraDF, None]:
    """ 
    Generate copy of spectrum_df with fragments filtered by minimum intensity. 
    
    Parameters:
        spectrum_df: SpectraDF object containing pandas.DataFrame.
        intensity_min: Minimum relative intensity a peak needs to exceed. If below, it will be filtered out.
    Returns:
        SpectraDF or None
    """

    output_df = spectra_df.get_data()
    if output_df is None or output_df.empty: # no data provided, return None
        return None
    selection_map = [x >= intensity_min for x in output_df["intensity"]]
    output_df = output_df.loc[selection_map]
    if output_df.empty: # no data left after filtering, return None
        return None
    output_df.reset_index(inplace=True, drop=True)
    return SpectraDF(output_df)


def generate_order_index(value_series : pd.Series) -> np.ndarray:
    """ Generate rank order List for data im value_series. 
    
    For each element in value_series, get the rank order for this element. Rank orders start at 0 and increment in 
    steps of 1 up to the number of unique instances in value_series -1.

    Parameters:
        value_series: pd.Series with rankable numeric data.
    Returns:
        pd.Series with rank order for each element in value_series.
    """
    rank_order_series = np.array(value_series.rank(method="dense"), dtype=np.int64)
    return rank_order_series

def list_to_string(input_list : List) -> str:
    """ 
    Helper function used in get_heatmap. Turns list into string joined by ' & '. 
    
    Parameters:
        input_list : List with any content that can be converted to str
    Returns:
        Single str with all list elements joined together by the separator ' & '
    """
    return ' & '.join(map(str, input_list))

def generate_hovertext_addon_labels(
        mass_to_charge_ratio_aggregate_series : pd.Series, 
        intensity_aggregate_series : pd.Series, 
        is_neutral_loss_series : pd.Series
        ) -> List[str]:
    """ 
    Helper function used in get_heatmap. Generates list of hovertext strings for each element in fragmap scatter.

    Parameters:
        mass_to_charge_ratio_aggregate_series:
        intensity_aggregate_series:
        is_neutral_loss_series:
    Returns:
        List[str] with hover-text addons.

    """

    hover_text_list = []
    for mz, intensity, is_loss in zip(mass_to_charge_ratio_aggregate_series.tolist(), 
        intensity_aggregate_series.tolist(), is_neutral_loss_series.tolist()):
        if not is_loss: # actual fragment data
            hover_text_list.append( (f"Original m/z value(s): {list_to_string(np.round(mz,3))}," 
                f" <br>Original intensity value(s): {list_to_string(np.round(intensity,3))}"))
        else: # inferred neutral loss data
            hover_text_list.append(f"Original loss value(s): {list_to_string(np.round(mz,3))}")
    return hover_text_list

def generate_neutral_loss_marker_shapes(
        mz_bin_index : pd.Series, 
        spec_index : pd.Series
        ) -> List[go.layout.Shape]:
    """ Helper Function for get_heatmap. Generate maker shapes for neutral loss visualization in Fragmap. """
    radius = 0.25 # Fixed value, must be below 0.5 to guarantee circle marker is contained within heatmap cell
    static_shape_setting_kwargs = {'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': '#882255', 'opacity' : 0.5}
    neutral_loss_marker_shapes = [go.layout.Shape(
        x0= x - radius, y0= y - radius, x1= x + radius, y1= y + radius, **static_shape_setting_kwargs) 
        for x,y in  zip(mz_bin_index.to_list(), spec_index.to_list())]
    return neutral_loss_marker_shapes

def generate_fragmap_figure_object(
        heatmap_trace, 
        neutral_loss_marker_shapes, 
        y_axis_tickvals, 
        y_axis_ticktext, 
        x_axis_tickvals, 
        x_axis_ticktext
        ) -> go.Figure:
    """ Helper function for get_heatmap(). Assembles fragmap figure from traces and axis information. """
    fragmap_figure = go.Figure(data=[heatmap_trace])
    fragmap_figure.update_layout(
        shapes=neutral_loss_marker_shapes, template="simple_white", 
        yaxis=dict(fixedrange=True, tickmode='array', tickvals=y_axis_tickvals, ticktext=y_axis_ticktext),
        xaxis=dict(tickmode='array', tickvals=x_axis_tickvals, ticktext=x_axis_ticktext, 
            spikesnap='hovered data', spikemode='across', spikethickness = 1, spikecolor="black"),
        xaxis_title="Binned mass to charge ratio (value is end of bin)", yaxis_title="Spectrum Identifier",
        margin = {"autoexpand":True, "b" : 10, "l":10, "r":10, "t":10})
    return fragmap_figure

def get_heatmap(spectra_df: Union[SpectraDF, None]):
    """ 
    Generate Heatmap Trace 
    """
    if SpectraDF is None:
        return None
    plot_df = spectra_df.get_data()

    
    # Add order indexes for mass-to-charge-ratio bins and spectrum indices.
    plot_df["bin_index"] = generate_order_index(plot_df["mass_to_charge_ratio"])
    plot_df["spec_index"] = generate_order_index(plot_df["spectrum_identifier"])

    # Generate hovertext addon labels with aggregation information.
    hover_text_addon = generate_hovertext_addon_labels(
        plot_df["mass_to_charge_ratio_aggregate_list"], 
        plot_df["intensity_aggregate_list"], 
        plot_df["is_neutral_loss"])
    
    # Generate x and y axis tick values and texts (mapping rank order to meaningful axis values / text)
    y_axis_tickvals = np.unique(plot_df["spec_index"])
    y_axis_ticktext = [f"Spectrum {elem}" for elem in np.unique(plot_df["spectrum_identifier"])]
    x_axis_tickvals = np.unique(plot_df["bin_index"])
    x_axis_ticktext = np.unique(plot_df["mass_to_charge_ratio"])

    

    # Generate fragment heatmap trace
    fragment_heatmap_trace = go.Heatmap(
        x=plot_df["bin_index"].to_list(), 
        y=plot_df["spec_index"].to_list(), 
        z=plot_df["intensity"].to_list(),
        text = hover_text_addon, 
        xgap=1, ygap=1, colorscale = "purp")

    # generates a neutral loss list of intensities with actual intensities set to NaN.
    # neutral loss traces and fragment traces must be equal length lists for the tick alignment to work
    neutral_z = [-1 if neutral_loss_bool else np.nan for neutral_loss_bool in plot_df["is_neutral_loss"]]
    neutral_loss_trace = go.Heatmap(
        x=plot_df["bin_index"].to_list(), 
        y=plot_df["spec_index"].to_list(), 
        z=  neutral_z,
        text = hover_text_addon, xgap=1, ygap=1, 
        colorscale = ["#C0C0C0", "#C0C0C0"], showscale=False, showlegend=False)

    #neutral_loss_marker_shapes = generate_neutral_loss_marker_shapes(
    #    plot_df_neutral_losses["bin_index"], plot_df_neutral_losses["spec_index"])

    fragmap_figure = go.Figure(data=[fragment_heatmap_trace])
    fragmap_figure.add_trace(neutral_loss_trace)
    #fragmap_figure = go.Figure(data=[fragment_heatmap_trace, neutral_loss_trace])
    #fragmap_figure.update_yaxes(
    #    fixedrange=True, tickmode='array', tickvals=y_axis_tickvals, ticktext=y_axis_ticktext)
    #fragmap_figure.update_xaxes(
    #    tickmode='array', tickvals=x_axis_tickvals, ticktext=x_axis_ticktext, 
    #    spikesnap='hovered data', spikemode='across', spikethickness = 1, spikecolor="black")
    fragmap_figure.update_layout(
        template="simple_white", 
        yaxis=dict(fixedrange=True, tickmode='array', tickvals=y_axis_tickvals, ticktext=y_axis_ticktext),
        xaxis=dict(tickmode='array', tickvals=x_axis_tickvals, ticktext=x_axis_ticktext, 
            spikesnap='hovered data', spikemode='across', spikethickness = 1, spikecolor="black"),
        yaxis2=dict(fixedrange=True, tickmode='array', tickvals=y_axis_tickvals, ticktext=y_axis_ticktext),
        xaxis2=dict(tickmode='array', tickvals=x_axis_tickvals, ticktext=x_axis_ticktext, 
            spikesnap='hovered data', spikemode='across', spikethickness = 1, spikecolor="black"),
        xaxis_title="Binned mass to charge ratio (value is end of bin)", yaxis_title="Spectrum Identifier",
        margin = {"autoexpand":True, "b" : 10, "l":10, "r":10, "t":10})

    #fragmap_figure = generate_fragmap_figure_object(
    #    fragment_heatmap_trace, neutral_loss_marker_shapes, y_axis_tickvals, y_axis_ticktext, x_axis_tickvals, 
    #    x_axis_ticktext)
    return fragmap_figure

def join_spectradf(
        spectrum_df : Union[SpectraDF, None], 
        losses_df : Union[SpectraDF, None]
        ) -> Union[SpectraDF, None]:
    """ Joins spectra and losses data frames together provided they are not both None. """
    if spectrum_df is not None:
        spectrum_pandas = spectrum_df.get_data()
    else:
        spectrum_pandas = None
    if losses_df is not None:
        losses_pandas = losses_df.get_data()
    else:
        losses_pandas = None
    if losses_pandas is None and spectrum_pandas is None:
        return None
    return SpectraDF(pd.concat([spectrum_pandas, losses_pandas]))

def generate_fragmap(
        spectrum_list : List[Spectrum], 
        relative_intensity_threshold : float, 
        prevalence_threshold : int,
        mass_to_charge_ratio_minimum : float, 
        mass_to_charge_ratio_maximum : float, 
        bin_map : List[float],
        top_k : int, 
        ) -> None:
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
    """
    # TODO: CONSIDER: prefilter spectra to size 200 before doing all data processing. Requires additional input?

    spectrum_list = [filter_spectrum_top_k_intensity_fragments(spectrum, top_k) for spectrum in spectrum_list]

    # Get binned spectra and binned neutral loss spectra
    binned_spectrum_list = [bin_spectrum(spectrum, bin_map) for spectrum in spectrum_list]
    neutral_loss_spectrum_list = [
        bin_spectrum(compute_neutral_loss_spectrum(spectrum), bin_map) for spectrum in spectrum_list]

    # construct spectra df and neutral loss df
    spectra_df = spectrum_list_to_pandas(binned_spectrum_list)
    losses_df = spectrum_list_to_pandas(neutral_loss_spectrum_list)
    
    # Apply filters for mz and intensity values
    filtered_spectra_df = generate_intensity_filtered_df(
        spectra_df, relative_intensity_threshold)
    filtered_spectra_df = generate_mz_range_filtered_df(
        spectra_df,  mz_min = mass_to_charge_ratio_minimum, mz_max = mass_to_charge_ratio_maximum)
    
    filtered_losses_df = generate_mz_range_filtered_df(
        losses_df, mz_min = mass_to_charge_ratio_minimum, mz_max = mass_to_charge_ratio_maximum)

    all_plot_data = join_spectradf(filtered_spectra_df, filtered_losses_df)
    all_plot_data = generate_prevalence_filtered_df(all_plot_data, n_min_occurrences=prevalence_threshold)

    fragmap = get_heatmap(all_plot_data)

    return fragmap
