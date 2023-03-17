from specxplore import data_transfer_cython
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
def generate_degree_colored_elements(sources, targets, values, threshold):
    """ 
    Generates degree elements and style sheet for coloring cytoscape nodes by their degree in plasma color scheme. 
    """
    # edge case: there are no edges at current threshold
    # edge case: min and max degree are the same, and no discrete color scale can be established, n_colors related
    print(sources[0:2], targets[0:2], values[0:2])
    tmp_values, tmp_sources, tmp_targets = data_transfer_cython.extract_edges_above_threshold(
        sources, targets, values, threshold)
    unique_nodes, node_counts = np.unique(np.concatenate([tmp_sources, tmp_targets]), return_counts= True)

    if len(unique_nodes) <= 1:
        ... # TDOD include edge case logic
    max_degree = np.max(node_counts)
    min_degree = np.min(node_counts)
    n_colors = 20
    n_bins = 20

    color_map = px.colors.sample_colorscale("plasma_r", [n/(n_colors -1) for n in range(n_colors)])
    degree_bins = np.linspace(min_degree, max_degree, num = n_bins, dtype=np.int)
    print("DEGREE BINS:", degree_bins)
    legend_fig = generate_plotly_bar_legend_for_colorscale(degree_bins, color_map)
    
    # generates color bin assignment indices for each unique nodes' degree
    color_bin_assignment_indices =  np.digitize(node_counts, degree_bins)-1

    print(np.array(color_map).shape, degree_bins.shape, color_bin_assignment_indices.shape)

    # color map is an array of size 100
    # color bin assignment indices is an array of size n with indices for color map
    color_assignment_array = np.array(color_map)[color_bin_assignment_indices]
    print(len(color_assignment_array))
    # create node style sheet:
    styles = []
    for idx in range(0, unique_nodes.size):
        styles.append({
            "selector":'#{}'.format(unique_nodes[idx]), 
            "style":{
                "background-color":color_assignment_array[idx], }
        })
    print(styles[0:2])
    
    return styles, legend_fig


def generate_plotly_bar_legend_for_colorscale(degrees_binned, colors):
    """
    Generates a ploty bar graph figure object with color to degree mapping to serve as legend for the cytoscape graph.
    """
    tickvals = np.linspace(np.min(degrees_binned), np.max(degrees_binned), num=5, dtype=np.int)
    y = np.repeat([1], len(degrees_binned))
    
    fig = go.Figure()
    for idx, col in enumerate(colors):
        fig.add_bar(x=[degrees_binned[idx]], y = [1], marker_color = col, showlegend = False, name=col)

    #fig = px.bar(
    #    x=degrees_binned, y = y, color=degrees_binned, 
    #    color_discrete_map = { value : color for value, color in zip(list(degrees_binned), list(colors))},
    #    labels = {"x" : "Node Degree"})
    fig.update_layout(barmode='group', bargap=0, bargroupgap=0, yaxis = {
        'showgrid': False, # thin lines in the background
        'zeroline': False, # thick line at x=0
        'visible': False,},  # numbers below
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(0,0,0,0)',  
        hovermode=False, 
        xaxis_title = "Node Degree",
        #xaxis = {
        #    "tickvals" : degrees_binned, #[min] + list(tickvals) + [max] 
        #},
        height = 100,
        margin = {"autoexpand":True, "b" : 30, "l":10, "r":10, "t":10}
    )
    fig.update_coloraxes(showscale=False)
    return fig