class specxplore_data:
    def __init__(self, spectra, source, target, values, class_dict, 
        sm_ms2deepscore,sm_modified_cosine,sm_spec2vec,
        idx, x, y):
        """ Data Container Class for specXplore. This structure is assumed
        by all dashboard panels functions as input. """
        self.spectra = spectra # list of spectra
        self.source = source # source array for edges
        self.target = target # target array for edges
        self.values = values # values array for edge cutoff
        self.class_dict = class_dict
        self.sm_ms2deepscore = sm_ms2deepscore
        self.sm_modified_cosine = sm_modified_cosine
        self.sm_spec2vec = sm_spec2vec

        self.node_idx = idx
        self.node_x = x
        self.node_y = y
    def __str__(self):
        """ Method to pretty print selected class summaries. """
        print("specXplore data object expected by dashboard modules.")
