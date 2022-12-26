import plotly.graph_objects as go
from scipy.cluster import hierarchy
import numpy as np
import itertools
import pandas as pd
from dash import html, dcc

# generate augmap currently requires both an edge list creation
# and a sm matrix

# --> the sm matrix is used primarily for optimal leaf odering. once leaf
#     ordering is achieved, matrices are reshuffled using the indexes and
#     the code progresses to construct various long format dfs.

# for the sizes considered, augmap is also always fast. 

# Last priority to fix this. keep required structures in place for now. 
# Consider: streamline code, numpy_load from disk the submatrices, improve 
# long format edge list construction via cythonize.

# TODO: [ ] "Load" pairwise similarity matrices using numpy on disk indexing 
#           (need to be saved accordingly)
# TODO: [ ] Create more dynamic data structures to allow between 1 and 5 
#           similarity scores. 
#       [ ] implement corresponding layering in heatmap visual and hover inf.



def generate_augmap(
    clust_selection, SM_MS2DEEPSCORE, SM_MODIFIED_COSINE, SM_SPEC2VEC, 
    threshold):

    ids_int = [int(elem) for elem in clust_selection]
    # Extract relevant subset  of nodes from sm matrices
    tmp_sm1 = SM_MS2DEEPSCORE[ids_int, :][:, ids_int]
    tmp_sm2 = SM_MODIFIED_COSINE[ids_int, :][:, ids_int]
    tmp_sm3 = SM_SPEC2VEC[ids_int, :][:, ids_int]

    # Create optimal ordering for heatmap based on tmp_sm1
    Z = hierarchy.ward(tmp_sm1) # hierarchical clustering
    index = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, tmp_sm1))
    
    # Reorder all data in line with optimal ordering
    tmp_sm1 = tmp_sm1[index,:][:,index]
    tmp_sm2 = tmp_sm2[index,:][:,index]
    tmp_sm3 = tmp_sm3[index,:][:,index]

    # Reorder ids as new index 
    ids_int = np.array(ids_int)
    ids_int = ids_int[index]
    ids  = [str(e) for e in ids_int]

    # Create long dfs
    all_possible_edges = list(
        itertools.combinations(np.arange(0, tmp_sm1.shape[1]), 2))
    edges =  [
        {'x' : elem[0], 'y' : elem[1], 'sim' : tmp_sm1[elem[0]][elem[1]]} 
        for elem in all_possible_edges]
    edges2 = [
        {'x' : elem[0], 'y' : elem[1], 'sim' : 1} 
        for elem in all_possible_edges 
        if tmp_sm2[elem[0]][elem[1]] > threshold]
    edges3 = [
        {'x' : elem[0], 'y' : elem[1], 'sim' : 1} 
        for elem in all_possible_edges 
        if tmp_sm3[elem[0]][elem[1]] > threshold]
    
    longdf = pd.DataFrame(edges)
    long_level2 = pd.DataFrame(edges2)
    long_level3 = pd.DataFrame(edges3)

    # Main heatmap trace
    trace = go.Heatmap(x=ids, y=ids, z = tmp_sm1,  type = 'heatmap', 
        colorscale = 'Viridis', zmin = 0, zmax = 1, xgap=1, ygap=1) # <-- might be mistake
    data = [trace]
    fig_ah = go.Figure(data = data)

    # Add augmentation layers
    r = 0.1
    xy = [[elem["x"], elem["y"]] for index, elem in long_level2.iterrows()]
    xy2 = [[elem["x"], elem["y"]] for index, elem in long_level3.iterrows()]
    kwargs = {'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': 'white'}
    points = [
        go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, **kwargs) 
        for x, y in xy]
    kwargs = {
        'type': 'rect', 'xref': 'x', 'yref': 'y', 
        'line_color': 'red', 'line_width' : 1}
    r = 0.2
    more_points = [
        go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, **kwargs) 
        for x, y in xy2]
    
    fig_ah.update_layout(shapes=points + more_points,            
        yaxis_nticks=tmp_sm1.shape[1],
        xaxis_nticks=tmp_sm1.shape[1],
        margin = {"autoexpand":True, "b" : 100, "l":0, "r":50, "t":0},
        title_text=
            'Augmap ms2deepscore scores with mcs (o) and spec2vec ([]).', 
        title_x=0.01, title_y=0.01,)

    panel = html.Div([
        dcc.Graph(id="augmented_heatmap", figure=fig_ah)
    ])
    return panel