# update cython module using: 
# python3 setup.py build_ext --inplace
# then run this script
# Takes as input: specxplore phophe datastructure
import pickle
import specxplore.specxplore_data
from specxplore.specxplore_data import specxplore_data
from specxplore import cython_utils as cu
import numpy as np
with open("testing/results/phophe_specxplore.pickle", 'rb') as handle:
    data = pickle.load(handle)

sources, targets, values = cu.construct_long_format_sim_arrays(data.ms2deepscore_sim)

print(sources[0:5], targets[0:5], values[0:5])


print(cu.extract_edges_above_threshold(sources, targets, values, 0.98))
print(cu.extract_selected_above_threshold(sources, targets, values, np.array([14,16,17,18]), 0.98))

v, s, t = cu.extract_edges_above_threshold(sources, targets, values, 0.85)
print(cu.creating_branching_dict_new(s, t, 17, 3))