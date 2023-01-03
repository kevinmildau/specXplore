# Developer Notes
# Add a toggle for viridis or the diverging colors augmap into the settings and here
# Viridis has the advantage of being constant to the threshold changes, while
# the diverging implementation works great in concert with high thresholds through
# clutter reduction.

import plotly.graph_objects as go
from scipy.cluster import hierarchy
import numpy as np
import itertools
import pandas as pd
from dash import html, dcc
import plotly.express as px

def _extract_sub_matrices(idx, sm1, sm2, sm3):
    # Extract relevant subset of spec_ids from sm matrices
    out_sm1 = sm1[idx, :][:, idx]
    out_sm2 = sm2[idx, :][:, idx]
    out_sm3 = sm3[idx, :][:, idx]
    return out_sm1, out_sm2, out_sm3

def _generate_optimal_leaf_ordering_index(sm):
    Z = hierarchy.ward(sm) # hierarchical clustering, ward linkage
    index = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, sm))
    return index

def _reoder_matrices(ordered_index, sm1, sm2, sm3):
    out_sm1 = sm1[ordered_index,:][:,ordered_index]
    out_sm2 = sm2[ordered_index,:][:,ordered_index]
    out_sm3 = sm3[ordered_index,:][:,ordered_index]
    return out_sm1, out_sm2, out_sm3

def _generate_edge_shapes(idx, sm1, sm2, threshold):
    kwargs1 = {
        'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': 'white'}
    r1 = 0.1
    kwargs2 = {
        'type': 'rect', 'xref': 'x', 'yref': 'y', 
        'line_color': 'red', 'line_width' : 1}
    r2 = 0.2

    all_possible_edges = list(
        itertools.combinations(np.arange(0, sm1.shape[1]), 2))
    edges1 = [
        [ idx[elem[0]], idx[elem[1]] ]
        for elem in all_possible_edges
        if sm1[elem[0]][elem[1]] >= threshold]
    edges2 = [
        [ idx[elem[0]], idx[elem[1]] ]
        for elem in all_possible_edges
        if sm2[elem[0]][elem[1]] >= threshold]
    shapes1 = [
        go.layout.Shape(x0=x-r1, y0=y-r1, x1=x+r1, y1=y+r1, **kwargs1) 
        for x, y in edges1]
    shapes2 = [
        go.layout.Shape(x0=x-r2, y0=y-r2, x1=x+r2, y1=y+r2, **kwargs2) 
        for x, y in edges2]
    return shapes1, shapes2

def _construct_redblue_diverging_coloscale(threshold):
    """Function creates a non-symmetric red-blue divergin color scale 
    in range 0 to 1, with breakpoint at the provided threshold."""
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

def generate_augmap(
    clust_selection, SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC, 
    threshold):
    """ Function constructs augmap figure object from provided data and 
    threshold settings. """

    idx = [int(elem) for elem in clust_selection]
    sm1, sm2, sm3 = _extract_sub_matrices(
        idx, SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC)
    ordered_index = _generate_optimal_leaf_ordering_index(sm1)
    sm1, sm2, sm3 = _reoder_matrices(ordered_index, sm1, sm2, sm3)

    # Reorder ids as new index, construct string id list
    idx = np.array(idx)
    idx = idx[ordered_index]
    ids  = [str(e) for e in idx]

    redblue_diverging = _construct_redblue_diverging_coloscale(threshold)

    # Main heatmap trace
    trace = go.Heatmap(
        x=ids, y=ids, z = sm1, type = 'heatmap', 
        customdata= np.dstack((sm2, sm3)),
        hovertemplate=(
            'X: %{x}<br>Y: %{y}<br>MS2DeepScore: %{z:.4f}<br>'
            'Mod.Cosine:%{customdata[0]:.4f}<br>'
            'Spec2Vec: %{customdata[1]:.4f}<extra></extra>'),
        #colorscale = 'Viridis',
        colorscale=redblue_diverging,
        zmin = 0, zmax = 1, xgap=1, ygap=1)
    data = [trace]
    fig_ah = go.Figure(data = data)
    
    shapes1, shapes2 = _generate_edge_shapes(
        np.arange(0, sm1.shape[0]), sm2, sm3, threshold)

    fig_ah.update_layout(
        shapes=shapes1+shapes2,            
        yaxis_nticks=sm1.shape[1],
        xaxis_nticks=sm1.shape[1],
        margin = {"autoexpand":True, "b" : 100, "l":0, "r":50, "t":0},
        title_text= 
            'Augmap ms2deepscore scores with mcs (o) and spec2vec ([]).', 
       title_x=0.01, title_y=0.01,)
    return fig_ah

def generate_augmap_panel(
    clust_selection, SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC, 
    threshold):
    """ Wrapper function to place augmap figure into dash compatible 
    container."""
    fig_ah = generate_augmap(
        clust_selection, SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC, 
        threshold)
    panel = html.Div([
        dcc.Graph(id="augmented_heatmap", figure=fig_ah)])
    return panel