from dataclasses import dataclass
import numpy as np

class specxplore_data:
  def __init__(
    self, ms2deepscore_sim, spec2vec_sim, cosine_sim, 
    tsne_df, class_table, clust_table, is_standard, spectra, mz, specxplore_id
    ):
    self.ms2deepscore_sim = ms2deepscore_sim
    self.spec2vec_sim = spec2vec_sim
    self.cosine_sim = cosine_sim
    self.tsne_df = tsne_df
    #tmp_class_table = class_table.merge(
    #  clust_table, 
    #  how = 'inner', 
    #  on='specxplore_id').drop(["specxplore_id"], axis = 1)
    #tmp_class_table.replace(np.nan, 'Unknown')
    #self.class_table = tmp_class_table
    self.class_table = class_table
    self.is_standard = is_standard
    self.spectra = spectra
    self.mz = mz # precursor mz values for each spectrum
    self.specxplore_id = specxplore_id

@dataclass
class Spectrum:
    mass_to_charge_ratios : np.ndarray #np.ndarray[int, np.double] # for python 3.9 and up
    precursor_mass_to_charge_ratio : np.double
    identifier : np.int64
    intensities : np.ndarray = None,
    def __post_init__(self):
        if self.intensities is None:
            self.intensities = np.repeat(np.nan, self.mass_to_charge_ratios.size)
        assert self.intensities.shape == self.mass_to_charge_ratios.shape, (
            "Intensities and mass to charge ratios must be equal length.")