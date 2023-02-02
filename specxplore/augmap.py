import plotly.graph_objects as go
from scipy.cluster import hierarchy
import numpy as np
import itertools
import pandas as pd
from dash import html, dcc
import plotly.express as px

def generate_optimal_leaf_ordering_index(similarity_matrix):
    """ Function generates optimal leaf ordering index for given similarity matrix. """
    linkage_matrix = hierarchy.ward(similarity_matrix) # hierarchical clustering, ward linkage
    index = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(linkage_matrix, similarity_matrix))
    return index

def extract_sub_matrix(idx, similarity_matrix):
    """ Extract relevant subset of spec_ids from similarity matrix. """
    out_similarity_matrix = similarity_matrix[idx, :][:, idx]
    return out_similarity_matrix

@DeprecationWarning
def extract_sub_matrices(idx, similarity_matrix_1, similarity_matrix_2, similarity_matrix_3):
    """ Extract relevant subset of spec_ids from sm matrices. """
    out_similarity_matrix_1 = similarity_matrix_1[idx, :][:, idx]
    out_similarity_matrix_2 = similarity_matrix_2[idx, :][:, idx]
    out_similarity_matrix_3 = similarity_matrix_3[idx, :][:, idx]
    return out_similarity_matrix_1, out_similarity_matrix_2, out_similarity_matrix_3

def reoder_matrix(ordered_index, similarity_matrix):
    """ Function reorders matrices according to ordered_index provided. """
    out_similarity_matrix = similarity_matrix[ordered_index,:][:,ordered_index]
    return out_similarity_matrix

@DeprecationWarning
def reoder_matrices(ordered_index, similarity_matrix_1, similarity_matrix_2, similarity_matrix_3):
    """ Function reorders matrices according to ordered_index provided. """
    out_similarity_matrix_1 = similarity_matrix_1[ordered_index,:][:,ordered_index]
    out_similarity_matrix_2 = similarity_matrix_2[ordered_index,:][:,ordered_index]
    out_similarity_matrix_3 = similarity_matrix_3[ordered_index,:][:,ordered_index]
    return out_similarity_matrix_1, out_similarity_matrix_2, out_similarity_matrix_3

def generate_edge_list(idx, all_possible_edges, similarity_matrix, threshold):
    """ Function generates edge list from pairwise similarity matrix and threshold while abiding by idx ordering. """
    # developer note
    # Input is a list of integer locations idx, a list of lists where each sublist contains two integer identifiers, 
    # a similarity matrix filtered to idx range, and a threshold for edge cutoffs. 
    edges = [
        [ idx[elem[0]], idx[elem[1]] ]
        for elem in all_possible_edges
        if similarity_matrix[elem[0]][elem[1]] >= threshold]
    return edges

def generate_shape_list(edges, size_modifier, shape_kwargs):
    """ Generates list with plotly shape object for each [x,y] pair in edges. """
    shapes = [
        go.layout.Shape(
            x0=x-size_modifier, y0=y-size_modifier, 
            x1=x+size_modifier, y1=y+size_modifier, 
            **shape_kwargs) 
        for x, y in edges]
    return shapes

def construct_circle_marker_shapes(idx, color, all_possible_edges, threshold, similarity_matrix):
    """ Constructs circle marker shape list for provided similarity matrix. """
    kwargs_circle_shape = {'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': color}
    radius_circles = 0.1
    edges = generate_edge_list(idx, all_possible_edges, similarity_matrix, threshold)
    shapes_circles = generate_shape_list(edges, radius_circles, kwargs_circle_shape)
    return shapes_circles

def construct_rectangle_marker_shapes(idx, color, all_possible_edges, threshold, similarity_matrix):
    """ Constructs rectangle marker shape list for provided similarity matrix. """
    kwargs_rectangle_shape = {'type': 'rect', 'xref': 'x', 'yref': 'y', 'line_color': color, 'line_width' : 2}
    width_rectangle = 0.2
    edges = generate_edge_list(idx, all_possible_edges, similarity_matrix, threshold)
    shapes_rectangles = generate_shape_list(edges, width_rectangle, kwargs_rectangle_shape)
    return shapes_rectangles

def generate_overlay_markers(idx, similarity_matrix_1, similarity_matrix_2, threshold, colorblind = False):
    """ Generates circle and rectangle shape lists for AugMap. """
    # Determine marker color with color or black and white for colorblind friendlier representation.
    if colorblind:
        color = "black"
    else:
        color = '#39FF14'
    all_possible_edges = list(itertools.combinations(np.arange(0, similarity_matrix_1.shape[1]), 2))
    # Construct circle markers for similarity matrix 1
    shapes_circles = construct_circle_marker_shapes(
        idx, color, all_possible_edges, threshold, similarity_matrix_1)
    # Construct rectangle markers for similarity matrix 2
    shapes_rectangles = construct_rectangle_marker_shapes(
        idx, color, all_possible_edges, threshold, similarity_matrix_2)
    return shapes_circles, shapes_rectangles

def construct_redblue_diverging_coloscale(threshold):
    """ Creates a non-symmetric red-blue divergin color scale in range 0 to 1, with breakpoint at the provided 
    threshold."""
    color_range = list(np.arange(0,1,0.01))
    closest_breakpoint = min(color_range, key=lambda x: abs(x - threshold))
    n_blues = int(closest_breakpoint * 100 - 1)
    n_reds = int(100 - (closest_breakpoint * 100) + 1)
    print(n_blues, n_reds, type(n_blues))
    blues = px.colors.sample_colorscale(
        "Blues_r", [n/(n_blues -1) for n in range(n_blues)])
    reds = px.colors.sample_colorscale(
        "Reds", [n/(n_reds -1) for n in range(n_reds)])
    redblue_diverging = blues + reds
    return(redblue_diverging)

def generate_heatmap_colorscale(threshold, colorblind = False):
    """ Creates colorscale for heatmap in range 0 to 1, either grayscale or diverging around threshold."""
    if colorblind:
        # 100 for increments of 1 across 0 to 1
        colorscale = px.colors.sample_colorscale("greys_r", [n/(100 -1) for n in range(100)]) 
    else:
        colorscale = construct_redblue_diverging_coloscale(threshold)
    return(colorscale)

def generate_heatmap_trace(ids, similarity_matrix_1, similarity_matrix_2, similarity_matrix_3, colorscale):
    """ Returns main heatmap trace for AugMap. """
    heatmap_trace = [go.Heatmap(
        x=ids, y=ids, z = similarity_matrix_1, type = 'heatmap', 
        customdata= np.dstack((similarity_matrix_2, similarity_matrix_3)),
        hovertemplate=(
            'X: %{x}<br>Y: %{y}<br>MS2DeepScore: %{z:.4f}<br>'
            'Mod.Cosine:%{customdata[0]:.4f}<br>'
            'Spec2Vec: %{customdata[1]:.4f}<extra></extra>'),
        colorscale=colorscale, zmin = 0, zmax = 1, xgap=1, ygap=1)]
    return heatmap_trace

def generate_augmap_graph(
    clust_selection, similarity_matrix_ms2ds, similarity_matrix_modified_cosine, similarity_matrix_s2v, 
    threshold, colorblind = False):
    """ Constructs augmap figure object from provided data and threshold settings. """
    
    # Convert string input to integer iloc
    idx = [int(elem) for elem in clust_selection]
    n_elements = len(idx)
    
    similarity_matrix_1 = extract_sub_matrix(idx, similarity_matrix_ms2ds)
    similarity_matrix_2 = extract_sub_matrix(idx, similarity_matrix_modified_cosine)
    similarity_matrix_3 = extract_sub_matrix(idx, similarity_matrix_s2v)

    ordered_index = generate_optimal_leaf_ordering_index(similarity_matrix_1)

    similarity_matrix_1 = reoder_matrix(ordered_index, similarity_matrix_1)
    similarity_matrix_2 = reoder_matrix(ordered_index, similarity_matrix_2)
    similarity_matrix_3 = reoder_matrix(ordered_index, similarity_matrix_3)

    # Reorder ids and idx according to optimal leaf ordering (computed above)
    idx = np.array(idx)
    idx = idx[ordered_index]
    ids  = [str(e) for e in idx]
    
    colorscale = generate_heatmap_colorscale(threshold)
    heatmap_trace = generate_heatmap_trace(
        ids, similarity_matrix_1, similarity_matrix_2, similarity_matrix_3, colorscale)
    
    shapes1, shapes2 = generate_overlay_markers(
        np.arange(0, n_elements), similarity_matrix_2, similarity_matrix_3, threshold, colorblind)
    
    augmap_figure = go.Figure(data = heatmap_trace)
    augmap_figure.update_layout(
        shapes=shapes1+shapes2, yaxis_nticks=n_elements, xaxis_nticks=n_elements,
        margin = {"autoexpand":True, "b" : 10, "l":10, "r":10, "t":10}, title_x=0.01, title_y=0.01,) 
    return augmap_figure


def generate_augmap_panel(
    spectrum_ids, similarity_matrix_ms2ds, similarity_matrix_modified_cosine, similarity_matrix_s2v, threshold):
    """ Class generators for augmap and places augmap figure into dash compatible container."""
    
    # Check whether spectrum_ids input is compatible with AugMap (not none, at least 2 ids)
    if not spectrum_ids or len(spectrum_ids) == 1:
        return [html.H6("Provide at least 2 spec_ids for AugMap.")]
    augmap_figure = generate_augmap_graph(
        spectrum_ids, similarity_matrix_ms2ds, similarity_matrix_modified_cosine, similarity_matrix_s2v, threshold)
    augmap_panel = html.Div([dcc.Graph(
            id="augmented_heatmap", figure=augmap_figure, 
            style={"width":"100%","height":"80vh", "border":"1px grey solid"})])
    return augmap_panel