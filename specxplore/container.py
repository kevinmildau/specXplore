class specXplore_data_container:
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