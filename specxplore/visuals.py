# Developer Notes

# --> check exact output of construct_gray_palette
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
