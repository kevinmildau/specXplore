import numpy as np
from functools import reduce

def standardize_array(array : np.ndarray):
    # mean centering and deviation scaling
    out = (array - np.mean(array)) / np.std(array)
    return out

def scale_array_to_minus1_plus1(array : np.ndarray) -> np.ndarray:
    """ Rescales array to lie between -1 and 1."""
    # Normalised [-1,1]
    out = 2.*(array - np.min(array))/np.ptp(array)-1
    return out

def initialize_cytoscape_graph_elements(tsne_df, selected_class_data, highlight_bool):
    n_nodes = tsne_df.shape[0]
    nodes = [{}] * n_nodes 
    for i in range(0, n_nodes):
        if highlight_bool[i] == True:
            highlight_entry = " is_highlighted"
        else:
            highlight_entry = ""
        nodes[i] = {
            'data':  dict(id = str(i)),
            'classes': str(selected_class_data[i]) + highlight_entry, #.replace('_',''), #  color_class[i],
            'position':{'x':tsne_df["x"].iloc[i], 'y':-tsne_df["y"].iloc[i]},   
        }
    return nodes

def compose_function(*func): 
    """ Generic function composer making use of functools reduce. 
    
    :param *func: Any number n of input functions to be composed.
    :returns: A new function object.

    Notes:
    Works well in conjunction with functools:partial, where functions can be composed using functions with partially
    filled arguments. This is especially useful for small processing pipelines that are locally defined. e.g:
    threshold_filter_07 = partial(threshold_filter_array, threshold = 0.7)
    extract_top_5 = partial(extract_top_from_array, top_number = 5)
    pipeline = compose(threshold_filter_07, extract_top_5)
    Where pipeline is now a function with one argument (array) based on partial functions with default arguments set.
    """
    def compose(f, g):
        return lambda x : f(g(x))   
    return reduce(compose, func, lambda x : x)