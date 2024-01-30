from specxplore import utils_cython
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


def generate_degree_colored_elements(sources, targets, values, threshold):
    """ 
    Generates degree elements and style sheet for coloring cytoscape nodes by their degree in plasma color scheme. 
    """
    # edge case: there are no edges at current threshold
    # edge case: min and max degree are the same, and no discrete color scale can be established, only one color needed
    _, tmp_sources, tmp_targets = utils_cython.extract_edges_above_threshold_from_descending_array(
        sources, targets, values, threshold)
    if tmp_sources.size >=1 and tmp_targets.size >=1:
        # Count how often each unique node occurs. in edge list. If once, there is an edge
        unique_nodes, node_counts = np.unique(np.concatenate([tmp_sources, tmp_targets]), return_counts= True)
        max_degree = np.max(node_counts)
        min_degree = np.min(node_counts)
        n_unique_degrees = np.unique(node_counts).size
    else:
        max_degree = 0
        min_degree = 0
        n_unique_degrees = 0

    case_no_edges_no_nonzero_degrees = n_unique_degrees == 0
    if case_no_edges_no_nonzero_degrees:
        node_styles = [], 
        legend_fig = []
        return node_styles, legend_fig

    elif min_degree == max_degree:
        # there is no color grade since all degrees are identical
        color_map = np.array(['rgb(13, 8, 135)'])
        degree_bins = np.array([str(min_degree)])
        legend_fig = generate_plotly_bar_legend_for_colorscale(degree_bins, color_map)
        color_assignment_array = np.repeat(['rgb(13, 8, 135)'], unique_nodes.size)
    else:
        n_colors = min(20, n_unique_degrees)
        n_bins = min(20, n_unique_degrees)
        color_map = px.colors.sample_colorscale("plasma_r", [n/(n_colors -1) for n in range(n_colors)])
        degree_bins_numeric = np.linspace(min_degree, max_degree, num = n_bins, dtype=np.int64)
        degree_bins = [str(element) for element in degree_bins_numeric]
        legend_fig = generate_plotly_bar_legend_for_colorscale(degree_bins, color_map)
        # generates color bin assignment indices for each unique nodes' degree
        color_bin_assignment_indices =  np.digitize(node_counts, degree_bins_numeric)-1
        # color map is an array of size 100
        # color bin assignment indices is an array of size n with indices for color map
        color_assignment_array = np.array(color_map)[color_bin_assignment_indices]

    # create node style sheet:
    node_styles = []
    for idx in range(0, unique_nodes.size):
        node_styles.append({
            "selector":'#{}'.format(unique_nodes[idx]), 
            "style":{
                "background-color":color_assignment_array[idx], }
        })
    return node_styles, legend_fig


def generate_plotly_bar_legend_for_colorscale(degrees_binned, colors):
    """
    Generates a ploty bar graph figure object with color to degree mapping to serve as legend for the cytoscape graph.
    """
    
    fig = go.Figure()
    for idx, col in enumerate(colors):
        fig.add_bar(x=[degrees_binned[idx]], y = [1], marker_color = col, showlegend = False, name=col)

    fig.update_layout(barmode='group', bargap=0, bargroupgap=0, yaxis = {
        'showgrid': False, # thin lines in the background
        'zeroline': False, # thick line at x=0
        'visible': False,},  # numbers below
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(0,0,0,0)',  
        hovermode=False, 
        xaxis_title = "Node Degree",
        xaxis = {
            "tickvals" : degrees_binned, #[min] + list(tickvals) + [max] 
        },
        height = 100,
        margin = {"autoexpand":True, "b" : 30, "l":10, "r":10, "t":10}
    )
    fig.update_coloraxes(showscale=False)
    return fig