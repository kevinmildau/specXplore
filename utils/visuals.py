import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import re

def construct_grey_palette(n_colors, white_buffer = 20, black_buffer = 5):
    """Construct a grey scale color palette. Construct is such that the very 
    faint white colors are removed via white_buffer number. In addition,
    the black_buffer is used to removes the darkest, non-distinguioshable
    dark gray colors."""
    n_colors = n_colors + white_buffer + black_buffer
    colors = px.colors.sample_colorscale("greys", [n/(n_colors -1) for n in range(n_colors)])[white_buffer-1:n_colors-black_buffer]
    return colors

def extract_hex_from_rgb_string(rgb_string):
    """ Extracts hex code from rgb string."""
    floats = re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", rgb_string)
    #print(floats)
    ints = [round(float(elem)) for elem in floats]
    #print(ints)
    ints = [255 if (elem > 255) else elem for elem in ints]
    ints = [0 if (elem < 0) else elem for elem in ints]
    #print(ints)
    ints = tuple(ints)
    return('#%02x%02x%02x' % ints)

if False:
    txt =  'rgb(230.01, 255.99999999, -0.6)' # too high nums, too low nums are fixed automatically.
    print(extract_hex_from_rgb_string(txt))

def create_color_dict(colors, cluster):
    unique_clusters = list(set(cluster))
    unique_clusters = [str(e) for e in unique_clusters]
    color_dict = {clust : colors[idx] for idx, clust in enumerate(unique_clusters)}
    return color_dict
