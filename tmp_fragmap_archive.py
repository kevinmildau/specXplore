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

