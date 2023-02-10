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




def generate_fragmap_old(id_list: [int], spec_list: [matchms.Spectrum], 
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
    spectra_long = spectrum_list_to_pandas(id_list=id_list, spec_list=spec_list)
    binned_spectra = bin_spectra(spectra=spectra_long, bins=bins)

    # Step 2: Filter Binned Spectra
    filter_binned_fragments_by_prevalence(binned_spectra=binned_spectra, n_bin_cutoff=n_bin_cutoff)
    remaining_bins = np.unique(binned_spectra["bin"].tolist())


    # Step 3-1: Calculate and Bin Neutral Losses
    neutral_losses = get_neutral_losses_data_frame(spec_list=spec_list)
    binned_neutral = bin_spectra(spectra=neutral_losses, bins=bins)

    # Step 3-2: Keep only those bins featured in the binned spectrum set
    binned_neutral.drop(labels=binned_neutral.loc[~binned_neutral["bin"].isin(remaining_bins)].index, inplace=True)
    binned_neutral.reset_index(inplace=True, drop=True)


    # THIS SHOULD BE IN THE BEGINNING TO LIMIT NUMBER OF FRAGMENTS CONSIDERED
    # Step 4-1: Filter Binned Spectra
    discarded_bins = filter_binned_spectra(
        spectra=binned_spectra, intensity_threshold=rel_intensity_threshold,
        prevalence_threshold=prevalence_threshold, mz_bounds=(mz_min, mz_max))

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
    
    index_bin_map = pd.cut(x=spectra_data_frame_copy["mass_to_charge_ratio"], bins=bins, labels=bins[0:-1], include_lowest=True)
    
    spectra_data_frame_copy.insert(loc=2, column="bin", value=index_bin_map, allow_duplicates=True)

    # Return Binned Data as long pandas dataframe
    return spectra_data_frame_copy


# THIS CAN BE REPLACED WITH MATCHMS FILTERS FOR MZ BOUNDS, INTENSITY AND MAX NUMBER OF PEAKS
# 3 lines of code.
def filter_binned_spectra(spectra: pd.DataFrame,
                          intensity_threshold: float,
                          prevalence_threshold: int,
                          mz_bounds: Tuple[float, float],
                          to_discard=None) -> List[int]:
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

# TODO: INCORPORATE INTO GENERATE HEATMAP FUNCTION
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
        "mass_to_charge_ratio": binned_spectra["mass_to_charge_ratio"].tolist(), # --> actual mz values for each x_axis thing; could be tuple
        "intensity": binned_spectra["intensity"].tolist()
    })
    # Return Mapped Dataset
    return new_data_frame
# TODO: INCORPORATE INTO GENERATE DATA FRAMES MODULES (PLOT RELEVANT DATA CONSTRUCTION)
# generate_y_axis_labels_for_specs
# input to get_spectrum_map should ??? contain actual spec_ids for y axis labelling.
# beware of non-iloc downstream effects
def get_spectrum_map(spectra: [int], root: int):
    assert root in spectra, (
        f"Error: Root Spectrum {root} not present in provided spectrum list."
    )

    
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

def filter_binned_fragments_by_prevalence(binned_spectra: pd.DataFrame, n_bin_cutoff: int):
    """
    Return copy of spectrum data frame with prevalence filter applied to fragments / bins.
    """
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
