from turtle import width
from dash import html, dcc
import plotly.express as px
import numpy as np
import pandas as pd
import dash
import dash_cytoscape as cyto
import plotly.graph_objects as go
from scipy.cluster import hierarchy
import itertools
import pickle

def gen_tab_1(data, tsne_df, color_dict):
    """Cluster Selection as Cytoscape in t-sne layout not yet implemented.
    Current state is a plotly express subplot.
    """
    if data != None:
        ids = [elem["customdata"][0] for elem in data["points"]]
        #print(ids)
        #print(tsne_df)
        tmp = tsne_df.iloc[ids, :]
        fig = px.scatter(tmp, x = "x", y = "y", color = "clust", custom_data=["id"], color_discrete_map= color_dict)
        fig.update_layout(clickmode='event+select') 
        fig.update_traces(marker=dict(size=12, opacity = 0.6))
        out = html.Div([
            html.H3('Cluster Selection View:'),
            dcc.Graph(figure=fig)
        ])
    else:
        out = html.Div([
            html.H3('Select Points to visualize subnet.'),
        ])
    return out

def gen_tab_2(data, nodes, edges, active_style_sheet, color_dict):
    """Tab 2 Generator - Cluster View with t-sne layout and edges in cytoscape.
    """
    if data != None:
        ids = [str(elem["customdata"][0]) for elem in data["points"]]
        my_set = set(ids)
        sel_edges = []
        sel_nodes = []
        sel_classes = set()
        for elem in edges:
            if (elem["data"]["source"] in my_set) and (elem["data"]["target"] in my_set):
                sel_edges.append(elem)
        for elem in nodes:
            if elem["data"]["id"] in my_set:
                sel_nodes.append(elem)
                sel_classes.add(elem["classes"])
        style_sheet = active_style_sheet + [{'selector': f".{clust}",
         'style': { 'background-color': f"{color_dict[clust]}"}} for clust in list(sel_classes)]
        out = html.Div([
            cyto.Cytoscape(
                id='cytoscape-tsne-subnet',
                layout={'name': 'preset'},
                elements= sel_edges + sel_nodes,
                stylesheet=style_sheet,
                boxSelectionEnabled=True,
                style={'width': '100%', 'height': '60vh'},
            ),
        ])
        return out
    else:
        out = html.Div([html.H3("Select Points to display t-sne subnet using dash cytoscape.")])
        return out


def gen_tab_3(data, nodes, edges, active_style_sheet, color_dict):
    """Tab 3 Generator - Cluster View with t-sne layout and edges in plotly.
    """

    #edges = [{'data' : {'id': str(elem[0]) + "-" + str(elem[1]), 
    #  'source': str(elem[0]),'target': str(elem[1])}} 
    #  for elem in all_possible_edges if (adj_m[elem[0]][elem[1]] != 0)]
    #nodes = [{'data': {'id': str(elem)}, 'label': 'Node ' + str(elem),
    #'position': {'x': x_coords[elem], 'y': -y_coords[elem]}, 'classes': cluster[elem]} # <-- Added -y_coord for correct orientation in line with t-sne
    #for elem in np.arange(0, n_nodes)]

    if data != None:
        ids = [str(elem["customdata"][0]) for elem in data["points"]]
        my_set = set(ids)
        node_x = []
        node_y = []
        node_c = []
        node_id = []
        node_dict = {}

        edge_x = []
        edge_y = []

        # TODO: Improve efficiency of this code. Current implementation is pragmatic and might not work well for larger datasets.
        # Create Node Graph
        for node in nodes:
            if node["data"]["id"] in my_set:
                node_x.append(node["position"]['x'])
                node_y.append(node["position"]['y'] * -1)
                node_c.append(node["classes"])
                node_id.append(node["data"]["id"])
                node_dict[node["data"]["id"]] = {"x" : node["position"]['x'], "y" : node["position"]['y'] * -1}
        #print(edges[0])
        for edge in edges:
            if edge["data"]["source"] in my_set and edge["data"]["target"] in my_set:
                edge_x.append(node_dict[edge["data"]["source"]]["x"]) # x0
                edge_x.append(node_dict[edge["data"]["target"]]["x"]) # x1
                edge_x.append(None)
                edge_y.append(node_dict[edge["data"]["source"]]["y"]) # y0
                edge_y.append(node_dict[edge["data"]["target"]]["y"]) # y1
                edge_y.append(None)


        fig = px.scatter(x = node_x, y = node_y, color = node_c, custom_data=[node_id], 
          color_discrete_map= color_dict)
        fig.update_layout(clickmode='event+select') 
        fig.update_traces(marker=dict(size=12, opacity = 0.8))
        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='rgba(0, 0, 0, 0.2)'), #rgba based opacity
            hoverinfo='none',
            mode='lines', name = "edges")

        fig.add_trace(edge_trace)
        out = html.Div([
            html.H3('Cluster Selection View:'),
            dcc.Graph(figure=fig)
        ])
        return out
    else:
        out = html.Div([html.H3("Select Points to display plotly express network in t-sne.")])
    return out

def gen_tab_4(data):
    """Tab 4 Generator - ego net ego layout generator"""
    out = html.Div([
        html.H3('Egonet with ego layout not yet included.'),
    ])
    return out


def gen_tab_5(data):
    """Tab 5 Generator - ego net dendrogram generator"""
    out = html.Div([
        html.H3('Egonet with dendrogram layout not yet included.'),
    ])
    return out

def gen_tab_6(data):
    """Tab 6 Generator - ego net t-SNE layout preserved."""
    out = html.Div([
        html.H3('Egonet t-SNE not yet included.')
    ])
    return out

def gen_tab_7(data, sm, sm2, sm3):
    """Tab 7 Generator - Augmented Heatmap for spectrum selection."""
    if data != None:
        ids_int = [int(elem["customdata"][0]) for elem in data["points"]]
        
        # Extract relevant subset  of nodes from sm matrices
        tmp_sm1 = sm[ids_int, :][:, ids_int]
        tmp_sm2 = sm2[ids_int, :][:, ids_int]
        tmp_sm3 = sm3[ids_int, :][:, ids_int]
        threshold = 0.7

        # Create optimal ordering for subset of nodes (index set)
        Z = hierarchy.ward(tmp_sm1) # hierarchical clustering
        index = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z, tmp_sm1))
        
        # Reorder vis_set according to clust_set index ordering
        tmp_sm1 = tmp_sm1[index,:][:,index]
        tmp_sm2 = tmp_sm2[index,:][:,index]
        tmp_sm3 = tmp_sm3[index,:][:,index]
        #print(ids_int)
        #print(list(index))
        # Reorder ids as new index 
        ids_int = np.array(ids_int)
        ids_int = ids_int[index]
        ids  = [str(e) for e in ids_int]

        # Create long dfs
        all_possible_edges = list(itertools.combinations(np.arange(0, tmp_sm1.shape[1]), 2))
        edges =  [{'x' : elem[0], 'y' : elem[1], 'sim' : tmp_sm1[elem[0]][elem[1]]} for elem in all_possible_edges]
        edges2 = [{'x' : elem[0], 'y' : elem[1], 'sim' : 1} for elem in all_possible_edges if tmp_sm2[elem[0]][elem[1]] > threshold]
        edges3 = [{'x' : elem[0], 'y' : elem[1], 'sim' : 1} for elem in all_possible_edges if tmp_sm3[elem[0]][elem[1]] > threshold]
        longdf = pd.DataFrame(edges)
        long_level2 = pd.DataFrame(edges2)
        long_level3 = pd.DataFrame(edges3)

        # Main heatmap trace
        trace = go.Heatmap(x=ids, y=ids, z = tmp_sm1,  type = 'heatmap', colorscale = 'Viridis', zmin = 0, zmax = 1, xgap=1, ygap=1) # <-- might be mistaked
        data = [trace]
        fig_ah = go.Figure(data = data)
        fig_ah.update_layout(width = 800, height = 800)
        # Add augmentation layers
        r = 0.1
        xy = [[elem["x"], elem["y"]] for index, elem in long_level2.iterrows()]
        xy2 = [[elem["x"], elem["y"]] for index, elem in long_level3.iterrows()]
        kwargs = {'type': 'circle', 'xref': 'x', 'yref': 'y', 'fillcolor': 'white'}
        points = [go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, **kwargs) for x, y in xy]
        kwargs = {'type': 'rect', 'xref': 'x', 'yref': 'y', 'line_color': 'red', 'line_width' : 1}
        r = 0.2
        more_points = [go.layout.Shape(x0=x-r, y0=y-r, x1=x+r, y1=y+r, **kwargs) for x, y in xy2]
        fig_ah.update_layout(shapes=points + more_points,            
            yaxis_nticks=tmp_sm1.shape[1],
            xaxis_nticks=tmp_sm1.shape[1])

        out = html.Div([
            html.H3('Heatmap for ms2deepscore scores with modifed cosine scores as circles and spec2vec as rectangles (threshold 0.7)'),
            dcc.Graph(id="augmented_heatmap", figure=fig_ah)
        ])
        return(out)
    else:
        out = html.Div([
            html.H3('Provide Point Selection for Augmented Heatmap.')
        ])
        return out


# Kev's upset plot visuals #####################################################

def spec_tab_1(data):
    if data != None:
        ids_int = [int(elem["customdata"][0]) for elem in data["points"]]
        binned_spectra = spectrum_filter_data(ids_int)[0]
        # Returns Upset plot featuring only selected bins as column
        upset_plot =threshold_bins_of_upset(0.01, 1, binned_spectra)
        out = html.Div([
            html.H3('Upset Plot'),
            dcc.Graph(id="upset-plot", figure=upset_plot)
        ])
        return out
    else: 
        out = html.Div([
                html.H3('Select Multiple Points to Visualize Spectral Overlap.'),
            ])
        return (out)

def spec_tab_2(data):
    if data != None:
        ids_int = [int(elem["customdata"][0]) for elem in data["points"]]
        spectrum_data = spectrum_filter_data(ids_int)[1]
        fig_spec = spectrum_plot(spectrum_data)
        out = html.Div([
            html.H3('Spectrum Plot'),
            dcc.Graph(id="upset-plot", figure=fig_spec)
        ])
        return out
    else: 
        out = html.Div([
                html.H3('Select Single Point to Visualize Spectrum.'),
            ])
        return (out)

## Convert Spectra to long format and bin for histogram

def spectrum_filter_data(sel_ids):
    """Helper function to extract and bin spectra based on numeric ids."""
    sel_set = set(sel_ids)
    file = open("data/cleaned_demo_data.pickle", 'rb')
    spectra = list(pickle.load(file))
    file.close()
    spectra = [spec for idx, spec in enumerate(spectra) if (idx in sel_set)]
    # Bin Spectra and Cast to Long Dataframe and Remove all values that are zero
    binned_spectra = [bin_spectrum(elem.mz, elem.intensities, bins=list(range(0, 1000 + 1, 1))) for elem in spectra]
    binned_spectra = matrix_to_long(matrix=np.array(object=binned_spectra))
    binned_spectra = binned_spectra.loc[binned_spectra['value'] != 0.0]
    #print(binned_spectra)
    spectrum_data = spectrum_list_to_long_data_frame([spectra[0]]) # <- for selecting only the first spectrum for single plot
    return (binned_spectra, spectrum_data)

################################################################################
################################################################################
################################################################################
################################################################################


# Henry's Upset plot functions #################################################


################################################################################
################################################################################
################################################################################
################################################################################
def spectrum_list_to_long_data_frame(spectra, indices=None):
    # Create a list of indices to include in final data frame
    indices = indices if indices else range(0, len(spectra))

    # Initialize empty data frame
    data_frame = pd.DataFrame()

    # Iterate over all indices specified or all indices in the data
    for index_index, index in enumerate(indices):
        # Extra data, create pandas data frame, and append to growing global data frame
        data = pd.DataFrame({
            "index": index,
            "m/z": spectra[index].mz,
            "intensity": spectra[index].intensities})
        data_frame = pd.concat([data_frame, data])

    # Return global data frame
    return data_frame


def matrix_to_long(matrix):
    # Cast matrix to pandas
    data_frame = pd.DataFrame(
        data=matrix,
        columns=list(range(0, matrix.shape[1])))

    # Add column of "names"
    data_frame.insert(
        loc=0,
        column='spectrum',
        value=list(range(0, matrix.shape[0])))

    # Cast wide to long
    data_frame = pd.melt(data_frame,
                         id_vars='spectrum',
                         var_name="bin",
                         ignore_index=True)

    # Sort by spectrum "name" and within that the bin
    data_frame.sort_values(["spectrum", "bin"],
                           ascending=[True, True],
                           inplace=True,
                           ignore_index=True)

    # Return long-formatted pandas dataframe
    return data_frame


def bin_spectrum(mz, intensity, bins):
    """ Function bins spectrum into specified bin ranges.

    Binning sums up the intensities of the joined fragments. Create bins using
    numpy.arange or numpy.linspace

    Parameters
    ----------------------------------------------------------------------------
    param1 : mz
        An array of mz values to be put into standardized bins.
    param2 : intensity
        An array of intensity values corresponding to the mz values.
        Must be equal in length to mz array.
    param3 : bins
        An array of breakpoints defining the bins [begin, step1, step2, ..., end]

    Returns
    ----------------------------------------------------------------------------
    array :
        An array of binned mz values with numeric data representing summed
        intensities for the bin.
    """
    assert len(mz) == len(intensity), "mz and intensity arrays must be equal length!"
    binned = np.histogram(mz, bins, weights = intensity)
    return binned[0]


def upset_heatmap(data_frame):
    # TODO: scatter plot implementation of same idea

    selected_bins = list(set(data_frame["bin"].tolist()))
    selected_bins.sort()

    #frames = go.Heatmap(x=data_frame['bin'].tolist(),
    #                    y=data_frame['spectrum'].tolist(),
    #                    z=[1] * data_frame.shape[0],
    #                    xgap=10,
    #                    ygap=10,
    #                    colorscale=["darkgrey", "darkgrey"])

    #print(selected_bins)
    heatmap = go.Heatmap(x=data_frame['bin'].tolist(),
                         y=data_frame['spectrum'].tolist(),
                         z=data_frame['value'].tolist(),
                         zmin=0,
                         zmax=1,
                         xgap=10,
                         ygap=10,
                         colorscale='dense')

    fig = go.Figure() # data=frames
    fig.add_trace(heatmap)

    fig.update_layout(template="simple_white")# , yaxis=dict(scaleanchor='x'))
    fig.update_xaxes(type='category', categoryarray=selected_bins)
    fig.update_yaxes(type='category')

    return fig


def upset_scatter(data_frame):

    trace = go.Scatter(x=data_frame["bin"],
                       y=data_frame["spectrum"],
                       mode="markers",
                       marker_symbol="circle-open",
                       marker=dict(color="red", size=10, line_width=1, line_color="red"))
    data = [trace]

    fig = go.Figure(data=data)
    fig.update_layout(template="simple_white",
                      yaxis=dict(scaleanchor='x'))
    fig.update_xaxes(type='category')
    fig.update_yaxes(type='category')

    return fig


def spectrum_plot(spectrum):
    # TODO: add option to freeze axes limits -> make inspection of multiple spectra easier
    # TODO: add mirror plot for comparing two spectra simultaneously

    fig = px.bar(data_frame=spectrum,
                 x="m/z",
                 y="intensity",
                 template="plotly_white")
    fig.update_traces(width=0.5)  # TODO: replace hard-coded size with range-driven setting instead
    return fig

def threshold_bins_of_upset(abundance_threshold, frequency_threshold, binned_data):
    # Initialize all list of bins
    selected_bins = list(set(binned_data["bin"].tolist()))
    bins_to_keep = []
    # Iterate over all binds in dataset
    for selected_bin in selected_bins:
        # Extract abundance values of all spectra for selected bin
        values = binned_data.loc[binned_data["bin"] == selected_bin]["value"].tolist()
        # If any abundance value exceed the set threshold -> keep bin
        if any([value >= abundance_threshold for value in values]) and (len(values) >= frequency_threshold):
            bins_to_keep.append(selected_bin)
    # Filter out non-selected bins from dataframe
    data_frame = binned_data.loc[binned_data['bin'].isin(bins_to_keep)]
    #print(data_frame)
    # Check how many bins exist per spectrum
    # Return Upset plot featuring only selected bins as column
    return upset_heatmap(data_frame)