import plotly.graph_objects as go
from scipy.cluster import hierarchy
import numpy as np
import itertools
from dash import html, dcc
import plotly.express as px
from typing import List
from specxplore.constants import UNICODE_DOT, UNICODE_SQUARE
def generate_optimal_leaf_ordering_index(similarity_matrix : np.ndarray):
    """ Function generates optimal leaf ordering index for given similarity matrix. """

    linkage_matrix = hierarchy.ward(similarity_matrix) # hierarchical clustering using ward linkage
    index = hierarchy.leaves_list(
        hierarchy.optimal_leaf_ordering(linkage_matrix, similarity_matrix)
    )
    return index

def extract_sub_matrix(idx : List[int], similarity_matrix : np.ndarray) -> np.ndarray:
    """ Extract relevant subset of spec_ids from similarity matrix. """

    out_similarity_matrix = similarity_matrix[idx, :][:, idx]
    return out_similarity_matrix

def reorder_matrix(ordered_index : List[int], similarity_matrix : np.ndarray) -> np.ndarray:
    """ Function reorders matrices according to ordered_index provided. """

    out_similarity_matrix = similarity_matrix[ordered_index,:][:,ordered_index]
    return out_similarity_matrix

def generate_edge_list(
        idx : List[int], 
        all_possible_edges : List[List[int]], 
        similarity_matrix : np.ndarray, 
        threshold : float
        ) -> List[List[int]]:
    """ 
    Function generates edge list from pairwise similarity matrix and threshold while abiding by idx ordering. 
    
    Developer Note:
    Input is a list of integer locations idx, a list of lists where each sublist contains two integer identifiers, 
    a similarity matrix filtered to idx range, and a threshold for edge cutoffs.
    """
    edges = [
        [ idx[elem[0]], idx[elem[1]] ]
        for elem in all_possible_edges
        if similarity_matrix[elem[0]][elem[1]] >= threshold
    ]
    return edges

def generate_shape_list(edges : List[List[int]], size_modifier : float, shape_kwargs : dict ):
    """ Generates list with plotly shape object for each [x,y] pair in edges. 
    
    Parameters:
    ----------
        edges:
        size_modifier:
        shape_kwargs: dictionary with  plotly.graph_objects.Layout.Shape properties
    """

    shapes = [
        go.layout.Shape(
            x0=x-size_modifier, y0=y-size_modifier, 
            x1=x+size_modifier, y1=y+size_modifier, 
            **shape_kwargs) 
        for x, y in edges
    ]
    return shapes

def construct_circle_marker_shapes(
        idx, 
        color, 
        all_possible_edges, 
        threshold, 
        similarity_matrix
        ):
    """ Constructs circle marker shape list for provided similarity matrix. """

    kwargs_circle_shape = {
        'type': 'circle', 
        'xref': 'x', 
        'yref': 'y', 
        'fillcolor': color, 
        'line_color' : color
    }
    radius_circles = 0.1
    edges = generate_edge_list(
        idx, 
        all_possible_edges, 
        similarity_matrix, 
        threshold
    )
    shapes_circles = generate_shape_list(
        edges, 
        radius_circles, 
        kwargs_circle_shape
    )
    return shapes_circles

def construct_rectangle_marker_shapes(
        idx, 
        color, 
        all_possible_edges, 
        threshold, 
        similarity_matrix
        ):
    """ Constructs rectangle marker shape list for provided similarity matrix. """

    kwargs_rectangle_shape = {
        'type': 'rect', 
        'xref': 'x', 
        'yref': 'y', 
        'line_color': color, 
        'line_width' : 2
    }
    width_rectangle = 0.2
    edges = generate_edge_list(
        idx, 
        all_possible_edges, 
        similarity_matrix, 
        threshold
    )
    shapes_rectangles = generate_shape_list(
        edges, 
        width_rectangle, 
        kwargs_rectangle_shape
    )
    return shapes_rectangles

def generate_overlay_markers(
        idx, 
        primary_score, 
        secondary_score, 
        threshold, 
        colorblind = False
        ):
    """ Generates circle and rectangle shape lists for AugMap. """
   
    # Determine marker color with color or black and white for colorblind friendlier representation.
    if colorblind:
        color = "magenta"
    else:
        color = '#27fa00'

    all_possible_edges = list(
        itertools.combinations(
            iterable = np.arange(0, primary_score.shape[1]), 
            r = 2
        )
    )
    # Construct circle markers for similarity matrix 1
    shapes_circles = construct_circle_marker_shapes(
        idx, color, all_possible_edges, threshold, primary_score)
    # Construct rectangle markers for similarity matrix 2
    shapes_rectangles = construct_rectangle_marker_shapes(
        idx, color, all_possible_edges, threshold, secondary_score)
    return shapes_circles, shapes_rectangles

def construct_redblue_diverging_coloscale(threshold):
    """ 
    Creates a non-symmetric red-blue divergin color scale in range 0 to 1, with breakpoint at the provided threshold.
    """
    
    color_range = list(np.arange(0,1,0.01))
    closest_breakpoint = min(color_range, key=lambda x: abs(x - threshold))
    n_blues = int(closest_breakpoint * 100 - 1)
    n_reds = int(100 - (closest_breakpoint * 100) + 1)
    blues = px.colors.sample_colorscale(
        "Blues_r", 
        [ n/(n_blues -1) for n in range(n_blues) ]
    )
    reds = px.colors.sample_colorscale(
        "Reds", [n/(n_reds -1) for n in range(n_reds)])
    redblue_diverging = blues + reds
    return(redblue_diverging)

def generate_heatmap_colorscale(threshold, colorblind = False):
    """ Creates colorscale for heatmap in range 0 to 1, either grayscale or diverging around threshold. """

    if colorblind:
        # 100 increments between 0 to 1
        colorscale = px.colors.sample_colorscale("greys", [n/(100 -1) for n in range(100)]) 
    else:
        colorscale = construct_redblue_diverging_coloscale(threshold)
    return(colorscale)

def generate_heatmap_trace(
        ids, 
        primary_score, 
        secondary_score, 
        tertiary_score, 
        colorscale,
        score_names
        ):
    """ Returns main heatmap trace for AugMap. """

    heatmap_trace = [
        go.Heatmap(
            x=ids, 
            y=ids, 
            z = primary_score, 
            type = 'heatmap', 
            customdata= np.dstack((secondary_score, tertiary_score)),
            hovertemplate = ''.join([
                    'X: %{x}<br>Y: %{y}<br>', 
                    score_names[0], # primary score
                    ': %{z:.4f}<br>',
                    UNICODE_DOT, 
                    score_names[1], # secondary score
                    ':%{customdata[0]:.4f}<br>',
                    UNICODE_SQUARE,
                    score_names[2], # tertiary score
                    ': %{customdata[1]:.4f}<extra></extra>'  
                ]
            ),
            colorscale=colorscale, 
            zmin = 0, 
            zmax = 1, 
            xgap=1, 
            ygap=1
        )
    ]
    return heatmap_trace

def generate_augmap_graph(
        clust_selection,
        primary_score, 
        secondary_score, 
        tertiary_score, 
        threshold, 
        score_names,
        colorblind = False,
        ):
    """ Constructs augmap figure object from provided data and threshold settings. """
    
    # Convert string input to integer iloc
    idx_iloc_list = [int(elem) for elem in clust_selection]
    n_elements = len(idx_iloc_list)
    
    # Extract similarity matrices for selection
    primary_score = extract_sub_matrix(idx_iloc_list, primary_score)
    secondary_score = extract_sub_matrix(idx_iloc_list, secondary_score)
    tertiary_score = extract_sub_matrix(idx_iloc_list, tertiary_score)

    # Generate optimal order index based on primary similarity matrix
    ordered_index = generate_optimal_leaf_ordering_index(primary_score)

    # Reorder similarity matrices according to optimal leaf ordering
    primary_score = reorder_matrix(ordered_index, primary_score)
    secondary_score = reorder_matrix(ordered_index, secondary_score)
    tertiary_score = reorder_matrix(ordered_index, tertiary_score)

    # Reorder ids and idx according to optimal leaf ordering (computed above)
    idx_iloc_array = np.array(idx_iloc_list)[ordered_index]
    ids_string_list  = [str(e) for e in idx_iloc_array]
    
    # Generate heathmap and joint hover trace
    colorscale = generate_heatmap_colorscale(threshold, colorblind)
    heatmap_trace = generate_heatmap_trace(
        ids_string_list, 
        primary_score, 
        secondary_score, 
        tertiary_score, 
        colorscale,
        score_names
    )
    
    # Generate overlay markers
    shapes1, shapes2 = generate_overlay_markers(
        np.arange(0, n_elements), 
        secondary_score, 
        tertiary_score, 
        threshold, colorblind
    )
    
    augmap_figure = go.Figure(data = heatmap_trace)
    augmap_figure.update_layout(
        shapes=shapes1+shapes2, 
        yaxis_nticks=n_elements, 
        xaxis_nticks=n_elements,
        margin = {
            "autoexpand":True, 
            "b" : 20, 
            "l":20, 
            "r":20, 
            "t":20
        }, 
        title_x=0.01, 
        title_y=0.01,) 
    return augmap_figure

def generate_augmap_panel(
        spectrum_ids, 
        primary_score, 
        secondary_score, 
        tertiary_score, 
        threshold, 
        colorblind_boolean,
        score_names
        ):
    """ Class generators for augmap and places augmap figure into dash compatible container."""
    
    # Check whether spectrum_ids input is compatible with AugMap (not none, at least 2 ids)
    if not spectrum_ids or len(spectrum_ids) == 1:
        return [ html.H6("Provide at least 2 spec_ids for AugMap.") ]
    augmap_figure = generate_augmap_graph(
        spectrum_ids, 
        primary_score, 
        secondary_score, 
        tertiary_score, 
        threshold,
        score_names,
        colorblind_boolean
    )
    augmap_panel = html.Div(
        children = [
            dcc.Graph(
                id="augmented_heatmap", figure=augmap_figure, 
                style={
                    "width":"95vh",
                    "height":"90vh", 
                    "border":"1px grey solid"
                }
            )
        ]
    )
    return augmap_panel