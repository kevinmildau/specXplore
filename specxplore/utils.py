import numpy as np
from functools import reduce
import re
import os

def scale_array_to_minus1_plus1(array : np.ndarray) -> np.ndarray:
    """ Rescales array to lie between -1 and 1."""
    out = 2.*(array - np.min(array))/np.ptp(array)-1
    return out

def save_numpy_array_to_file (array : np.ndarray, filepath : str) -> None:
    """ Saves numpy array to text file with extension .npy """
    np.save(os.path.join(filepath, ".npy"), array, allow_pickle=False)
    return None

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


def extract_hex_from_rgb_string(rgb_string):
    """ Extracts hex code from rgb string.
    
    Function forces any rgb values extracted to be within [0 255] range. 

    Args
    ------
    rgb_string:
        An rgb color string.

    Returns
    ------
    output:
        hex color string corresponding to input rgb string.

    Example:
    --------
    Numbers out of range are fixed automatically to be within range.

    txt =  'rgb(230.01, 255.99999999, -0.6)'

    print(extract_hex_from_rgb_string(txt))
    """
    floats = re.findall(
        "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", 
        rgb_string
    )
    ints = [round(float(elem)) for elem in floats]
    ints = [255 if (elem > 255) else elem for elem in ints]
    ints = [0 if (elem < 0) else elem for elem in ints]
    ints = tuple(ints)
    return('#%02x%02x%02x' % ints)


def update_expand_level(new_expand_level):
    """ Function updates expand level and placeholder in line with input.

    Expand levels define the number of branching out events for the 
    ego-network represenation. The default value is 1, for all direct 
    connections to the root / ego-node. Large values or values below 1 are
    ill suited for this representation and automatically rejected. 
    A value of 0 would lead to no connections and defeat the point of the
    visualization. Negative values make no sense. Large values will connect
    increasingly irrelevant spectra, or lead to complete graphs depending on
    threshold settings. Sensible defaults are [1 6].
    
    Args / Parameters
    ------
    new_expand_level: 
        New expand level; integer >=1 and should be a small number in line
        with lower and upper limit settings.

    Returns
    ------
    output 0:
        A new expand level setting as integer.
    output 1:
        A new placeholder text for expand level text input box.
    """
    lower_limit = 1
    upper_limit = 6
    if (new_expand_level 
        and new_expand_level >= lower_limit 
        and new_expand_level <= upper_limit
        and isinstance(new_expand_level, int)
        ):
        new_placeholder = (
            f'Expand Level {lower_limit} =< thr <= {upper_limit},'
            f'current: {new_expand_level}'
        )
        return new_expand_level, new_placeholder
    else:
        default_expand_level = 1
        default_placeholder = (
            f'Expand Level {lower_limit} =< thr <= {upper_limit},'
        )
        return default_expand_level,  default_placeholder

def update_threshold(new_threshold):
    """ Function updates threshold level and placeholder in line with input.
    
    Args:
    new_threshold: 
        New threshold; float < 1 and > 0.

    Returns:
    output 0:
        A new threshold setting, float.
    output 1:
        A new placeholder text for threshold text input box.
    """
    if (new_threshold 
        and new_threshold < 1 and new_threshold > 0
        and isinstance(new_threshold, float)
        ):
        new_placeholder = (
            'Threshold 0 < thr < 1,' 
            f'current: {new_threshold}'
        )
        return new_threshold, new_placeholder
    else:
        default_threshold = 0.9
        default_placeholder = "Threshold 0 < thr < 1, def. 0.9"
        return default_threshold,  default_placeholder


def update_max_degree(new_max_degree):
    """ Function updates maximum node degree and placeholder in line with input.
    
    Args / Parameters
    ------
    new_max_degree: 
        New maximum node degree; integer >=1 and <= 9999

    Returns
    ------
    output 0:
        A new expand level setting as integer.
    output 1:
        A new placeholder text for expand level text input box.
    """
    lower_limit = 1
    upper_limit = 9999
    if (
        new_max_degree 
        and new_max_degree >= lower_limit 
        and new_max_degree <= upper_limit
        and isinstance(new_max_degree, int)
        ):
        new_placeholder = (
            f'Maximum Node Degree {lower_limit} =< thr <= {upper_limit},'
            f'current: {new_max_degree}'
        )
        return new_max_degree, new_placeholder
    else:
        default_expand_level = 9999
        default_placeholder = (
            f'Maximum Node Degree {lower_limit} =< thr <= {upper_limit},'
        )
        return default_expand_level,  default_placeholder