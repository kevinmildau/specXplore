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