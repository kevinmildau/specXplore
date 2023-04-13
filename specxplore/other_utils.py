import numpy as np

def standardize_array(array : np.ndarray):
    # mean centering and deviation scaling
    out = (array - np.mean(array)) / np.std(array)
    return out

def scale_array_to_minus1_plus1(array : np.ndarray) -> np.ndarray:
    """ Rescales array to lie between -1 and 1."""
    # Normalised [-1,1]
    out = 2.*(array - np.min(array))/np.ptp(array)-1
    return out

def initialize_cytoscape_graph_elements(tsne_df, selected_class_data, is_standard):
    n_nodes = tsne_df.shape[0]
    nodes = [{}] * n_nodes 
    for i in range(0, n_nodes):
        if is_standard[i] == True:
            standard_entry = " is_standard"
        else:
            standard_entry = ""
        nodes[i] = {
            'data':  dict(id = str(i)),
            'classes': str(selected_class_data[i]) + standard_entry, #.replace('_',''), #  color_class[i],
            'position':{'x':tsne_df["x"].iloc[i], 'y':-tsne_df["y"].iloc[i]},   
        }
    return nodes