import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import re

def construct_grey_palette(n_colors, white_buffer = 20, black_buffer = 5):
    """Constructs a grey scale color palette. 
    
    The color palette is constructed such that the first n = white_buffer white 
    colors are removed since they are to faint to distinguish from one another. 
    Similarly, the black_buffer is used to removes the darkest, 
    non-distinguishable shades of gray.
    
    Parameters:
    ------
    n_colors:
        The number of colors to be generated.
    white_buffer:
        The number of light gray shades to remove.
    black_buffer
        The number of dark gray shades to remove.

    Returns:
    ------
    colors:
        List of color strings.
    """
    n_colors = n_colors + white_buffer + black_buffer
    colors = px.colors.sample_colorscale(
        "greys", [n/(n_colors -1) for n in range(n_colors)])
    colors = colors[white_buffer-1:n_colors-black_buffer]
    return colors

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

    """
    floats = re.findall(
        "[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", rgb_string)
    ints = [round(float(elem)) for elem in floats]
    ints = [255 if (elem > 255) else elem for elem in ints]
    ints = [0 if (elem < 0) else elem for elem in ints]
    ints = tuple(ints)
    return('#%02x%02x%02x' % ints)

if False:
    # too high nums, too low nums are fixed automatically.
    txt =  'rgb(230.01, 255.99999999, -0.6)'
    print(extract_hex_from_rgb_string(txt))

def create_color_dict(colors, cluster):
    """ Creates color dictionary with grayscale color for each unique cluster.

    Args
    ------
    colors:
        A list of color strings (hex)
    cluster:
        A list of cluster identifiers (int or str). The list may be unique
        identifiers or a list of cluster assignments to be turned into unique
        assignments by the function.
    Returns
    ------
    output:
        A color dictionary where each unique cluster is a string key with the
        corresponding value being a hex code color: {clust_key : color_string}.

    """
    unique_clusters = list(set(cluster))
    unique_clusters = [str(e) for e in unique_clusters]
    color_dict = {
        clust : colors[idx] 
        for idx, clust in enumerate(unique_clusters)}
    return color_dict

def extract_identifiers_from_plotly_selection(plotly_selection_data):
    """ Function extracts custom_data id's from a provided point selection 
    dictionary. 
        
    Args / Parameters
    ------
    plotly_selection_data:
        Selection data as returned by plotly (json format). spec_ids are 
        assumed too be stored inside the 'customdata' component of the 
        selection data at index 0.

    Returns
    ------
    output:
        List of spec_ids corresponding to the selected scatter points.
    
    """
    if plotly_selection_data != None:
        selected_ids = [
            elem["customdata"][0] for elem in plotly_selection_data["points"]]
    else:
        selected_ids = []
    return selected_ids

def update_class(selected_class, class_dict):
    """ Function updates selected_class_data and color_dict elements in line
    with provided selected_class.
        
    Args
    ------
    selected_class:
        String identifying the selected class. The string should be a key in 
        class_dict.
    class_dict:
        A dictionary with class_string keys and corresponding class assignment
        list of each spec_id: {class_string : class_assignment_list}. The class
        assignment list is ordered to correspond to spec_id idx ordering.

    Returns
    ------
    output 0:
        selected_class_data, a list of class assignments for each spec_id 
        in original idx ordering.
    output 1:
        A color dictionary where each unique cluster is a string key with the
        corresponding value being a hex code color: {clust_key : color_string}.
    """
    white_buffer = 20 # --> see construct_grey_palette() 
    selected_class_data = class_dict[selected_class]
    selected_class_data = [
        re.sub('[^A-Za-z0-9]+', '_', elem) for elem in selected_class_data]
    n_colors = len(set(selected_class_data))
    colors = construct_grey_palette(n_colors, white_buffer)
    color_dict = create_color_dict(colors, selected_class_data)
    return selected_class_data, color_dict

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
        and isinstance(new_expand_level, int)):
        new_placeholder = (
            f'Expand Level {lower_limit} =< thr <= {upper_limit},'
            f'current: {new_expand_level}')
        return new_expand_level, new_placeholder
    else:
        default_expand_level = 1
        default_placeholder = (
            f'Expand Level {lower_limit} =< thr <= {upper_limit},')
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
        and isinstance(new_threshold, float)):
        new_placeholder = (
            'Threshold 0 < thr < 1,' 
            f'current: {new_threshold}')
        return new_threshold, new_placeholder
    else:
        default_threshold = 0.9
        default_placeholder = "Threshold 0 < thr < 1, def. 0.9"
        return default_threshold,  default_placeholder