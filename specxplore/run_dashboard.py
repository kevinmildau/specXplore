from dash import dcc, html, ctx, dash_table, Dash
from dash.dependencies import Input, Output, State

import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto

import specxplore
from specxplore import egonet, augmap, clustnet, fragmap, data_transfer, specplot, degree_map, other_utils
import specxplore.datastructures
from specxplore.constants import COLORS

import pickle

import plotly.graph_objects as go

import os

import numpy as np


from typing import Union


specxplore_input_file = '/Users/kevinmildau/Dropbox/univie/Project embedding a molecular network/development/specxplore-illustrative-examples/data/data_wheat_output/specxplore_wheat.pickle'
with open(specxplore_input_file, 'rb') as handle:
    GLOBAL_DATA = pickle.load(handle) 
    GLOBAL_DATA.scale_coordinate_system(400)
####################################################################################################################
# All settings panel


####################################################################################################################
# All settings panel
settings_panel = dbc.Offcanvas(
    children=[
        html.B("Set Edge Threshold:"),
        html.P("A value between 0 and 1 (excluded)"),
        dcc.Input(
            id="edge_threshold_input_id", 
            type="number", 
            debounce=True, 
            placeholder="Threshold 0 < thr < 1, def. 0.9", 
            style={"width" : "100%"}
        ),
        html.Br(),
        html.Br(),
        html.Div(
            children=[
                dcc.Graph(
                    id = "edge-histogram-figure", 
                    figure = [],  
                    style={"width":"100%","height":"30vh"}
                )
            ]
        ),
        html.Br(),
        html.B("Set maximum node degree:"),
        html.P("A value between 1 and 9999. The lower, the fewer edges are allowed for each node."),
        dcc.Input(
            id="input-maximum-number-of-nodes", 
            type="number", 
            debounce=True, 
            placeholder="1 <= Value <= 9999, def. 9999", 
            style={"width" : "100%"}
        ),
        html.Br(),
        html.B("Higlight Class(es) by color:"),
        dcc.Dropdown(
            id='dropdown_classes_to_be_highlighted' , 
            multi=True, 
            clearable=False, 
            options=[],
            value=[]
        ),
        html.Br(),
        html.B("Set Hop Distance for EgoNet:"),
        html.P("A value between 1 and 5. Controls connection branching in EgoNet."),
        dcc.Input(
            id="expand_level_input", 
            type="number", 
            debounce=True, 
            placeholder="Value between 1 < exp.lvl. < 5, def. 1", 
            style={"width" : "100%"}
        ),
        html.Br(),
        html.B("Select Class Level:"),
        dcc.Dropdown( 
            id='select_class_level_dropdown' , 
            multi=False, 
            clearable=False, 
            options= GLOBAL_DATA.available_classes, 
            value=GLOBAL_DATA.available_classes[0]
        ),
        html.Br(),
        html.B("Toggle colorblind friendly for Augmap:"),
        daq.BooleanSwitch(
            id='augmap_colorblind_friendly_switch', 
            on=False
        ),
        html.Br(),
        html.B("Set top-K for fragmap"),
        html.P(
            children=(
                "A value between 1 and 9999. This value is used to restrict the number of"
                " fragments displayed in fragmap. "
                "If the number of fragments exceeds top-K, only the top-K highest intensity"
                " fragments are displayed " 
                "alongside the corresponding neutral losses."
            )
        ),
        dcc.Input(
            id="top_k_fragments", type="number", 
            debounce=True, value=25, placeholder = "25",
            style={"width" : "100%"}
        ),
        html.Br(),
        html.B("Set Visal Scale:"),
        html.P(
            children=(
                "A value between 0 excluded and 2000. This value is used to expand the t-SNE layout to be"
                " larger or smaller."
            )
        ),
        dcc.Input(
            id="scaler_id", 
            type="number", 
            debounce=True, 
            value=400, 
            placeholder = "400",
            style={"width" : "100%"}
        ),
        html.Br(),
        html.B("Session Data Location"),
        html.P("Provide path to a valid specXplore.pickle object."),
        html.Div(
            children=[
                dcc.Input(
                    id="upload-data", 
                    type="text", 
                    placeholder=None, 
                    debounce=True)
            ]
        ),
        # Multi-line break to make sure that dcc dropdown interactions don't cause resizing issues
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),
    ],
    id="offcanvas-settings",
    title="specXplore settings",
    is_open=False)

####################################################################################################################
# Focus selection panel
selection_focus_panel = dbc.Offcanvas(
    children=[
        html.P(
            children = (
                "All spectrum ids selected for focused analysis in network views."
                "\n Selection can be modified here."
            )
        ),
        dcc.Dropdown(
            id='spectrum-iloc-focus-dropdown', 
            multi=True, 
            style={'width': '90%', 'font-size': "100%"}, 
            options=GLOBAL_DATA.get_spectrum_iloc_list())
    ],
    id="offcanvas-focus",
    placement="end",
    title="Selection Panel",
    is_open=False)

####################################################################################################################
# Button array on left side of plot

button_height = "9vh"
button_style_text = {
    'width': '98%', 
    'height': button_height, 
    "fontSize": "11px", 
    "textAlign": "center", 
    "border":"1px black solid",
    'backgroundColor' : '#8B008B'
}

control_button_group = [
    dbc.ButtonGroup(
        children=[
            dbc.Button('⚙ Settings', id="btn-open-settings", style = button_style_text),
            dbc.Button('View Selection', id="btn-open-focus", style = button_style_text),
            dbc.Button('Show Node Degree ', id='btn-run-degree', n_clicks=0, style = button_style_text),
            dbc.Button('EgoNet ', id='btn-run-egonet', n_clicks=0, style = button_style_text),
            dbc.Button('Show Edges for Selection', id="btn-run-clustnet", n_clicks=0, style = button_style_text),  
            dbc.Button('Augmap', id="btn-push-augmap", n_clicks=0, style = button_style_text),
            dbc.Button('Spectrum Plot(s)', id='btn_push_spectrum', n_clicks=0, style = button_style_text),
            dbc.Button('Fragmap', id="btn_push_fragmap", n_clicks=0, style = button_style_text),
            dbc.Button('Metadata Table', id="btn_push_meta", n_clicks=0, style = button_style_text)
        ],
        vertical=True, 
        style = {"width" : "100%"}
    )
]

####################################################################################################################
# good themes: VAPOR, UNITED, SKETCHY is the coolest by far ; see more:  https://bootswatch.com/
app=Dash(external_stylesheets=[dbc.themes.UNITED])

app.layout=html.Div(
    children=[
        # ROW 1: Title & hover response
        dbc.Row(
            children = [
                dbc.Col(
                    children = [
                        html.H1(
                            children = [
                                html.B("specXplore prototype")
                            ], 
                            style={"margin-bottom": "0em", "font-size" : "20pt"}
                        )
                    ], 
                    width=6),
            ]
        ),
        dbc.Row(
            children = [
                dbc.Col(control_button_group, width = 2),
                dbc.Col(
                    children = [
                        cyto.Cytoscape( 
                            id='cytoscape-tsne', 
                            elements=GLOBAL_DATA.initial_node_elements, 
                            stylesheet = GLOBAL_DATA.initial_style,
                            layout={
                                'name':'preset', 
                                'animate':False, 
                                'fit': False}, 
                            boxSelectionEnabled=True,
                            style={
                                'width':'100%', 
                                'height':'80vh', 
                                "border":"1px black solid", 
                                "bg":"#feeff4", 
                                'minZoom':0.1, 
                                'maxZoom':2})
                    ], 
                    width=10
                ) 
            ], 
            style={"margin": "0px"}, 
            className="g-0"),
        settings_panel,
        selection_focus_panel,
        dcc.Store(id='session_data_update_trigger', data = None),
        dcc.Store(id="edge_threshold", data=0.9),
        dcc.Store(id="expand_level", data=int(1)),
        dcc.Store(id='maximum-number-of-nodes', data = int(9999)),
        dcc.Store(id="selected_class_level_store", data= GLOBAL_DATA.available_classes[0]),
        dcc.Store(id='selected_class_level_assignments_store', data=[GLOBAL_DATA.available_classes[0]]),
        dcc.Store(id='node_elements_store', data = GLOBAL_DATA.initial_node_elements),
        dcc.Store(id='colorblind_friendly_boolean_store', data = False),
        #html.Br(),
        dbc.Row(
            children = [
                dbc.Col(
                    children = [
                        html.Tbody("Hover Info: placeholder for cyto hover information")
                    ], 
                    id = "hover-info-panel",
                    style={"font-size" : "8pt", "border":"1px black solid"}),
                dbc.Col(
                    children = [
                        dcc.Markdown(
                            id="warning-messages-panel1", 
                            style={"font-size" : "8pt", "border":"1px black solid"}),
                        dcc.Markdown(
                            id="warning-messages-panel2", 
                            style={"font-size" : "8pt", "border":"1px black solid"})
                    ], 
                    width=4), 
                dbc.Col(
                    children = [
                        html.Div(id="legend_panel", style={"width":"100%", "border":"1px black solid"})
                    ], 
                    width=8)
                ], 
            className="g-0"), 
        dbc.Row(
            children = [
                dbc.Col(
                    children = [
                        html.Div(
                            id="details_panel", 
                            style={"width":"100%", "height":"90vh", "border":"1px black solid"})
                    ], 
                    width=12)
            ], 
            className="g-0"
        ),
    ], 
    style={"width" : "100%"},
)

####################################################################################################################
# cytoscape update callbacks

@app.callback(
    Output('cytoscape-tsne', 'elements'),
    Output('cytoscape-tsne', 'stylesheet'),
    Output('cytoscape-tsne', 'zoom'),
    Output('cytoscape-tsne', 'pan'),
    Output('legend_panel', 'children'),
    Output('warning-messages-panel1', 'children'),
    Input('btn-run-egonet', 'n_clicks'),
    Input('btn-run-clustnet', 'n_clicks'),
    Input('btn-run-degree', 'n_clicks'),
    Input('spectrum-iloc-focus-dropdown', 'value'), 
    Input('dropdown_classes_to_be_highlighted', 'value'),
    Input('selected_class_level_assignments_store', "data"),
    Input('node_elements_store', 'data'),
    Input('session_data_update_trigger', 'data'),
    State("maximum-number-of-nodes", "data"),
    State("edge_threshold", "data"),
    State("expand_level", "data"),
    State('cytoscape-tsne', 'zoom'),
    State('cytoscape-tsne', 'pan'),
    State('cytoscape-tsne', 'elements'),   # <-- for persistency
    State('cytoscape-tsne', 'stylesheet'), # <-- for persistency
    prevent_initial_call=True)
def cytoscape_trigger(
    _n_clicks1, 
    _n_clicks2,
    _n_clicks3,
    spec_id_selection, 
    classes_to_be_highlighted,
    all_class_level_assignments, 
    node_elements_from_store,
    none_data_trigger, 
    max_edges_per_node,
    threshold, 
    expand_level, 
    zoom_level, 
    pan_location,
    previous_elements,
    previous_stylesheet):
    """ Global overview interactivty and updating handling function. This function determines the updating sequence
    to be performed depending on the reactive element that triggered the function."""

    btn = ctx.triggered_id
    max_colors = 8
    max_edges_clustnet = 2500 # make an input state
    max_edges_egonet = 2500 # make an input state, make it actually used

    legend_panel = [] # initialize to empty list. Only get filled if degree node selected
    warning_messages = "" # initialize to empty string. Only gets expanded if warnings necessary.
    styles = GLOBAL_DATA.initial_style

    elements = node_elements_from_store

    case_highlight_classes =  btn == 'dropdown_classes_to_be_highlighted' and classes_to_be_highlighted
    case_too_many_classes_to_be_highlighted = (len(classes_to_be_highlighted) > max_colors)

    if case_highlight_classes:
        # define style color update if trigger is class highlight dropdown
        tmp_colors = [{
            'selector' : f".{str(elem)}", 'style' : {"background-color" : COLORS[idx]}} 
            for idx, elem in enumerate(classes_to_be_highlighted)]
        styles = GLOBAL_DATA.initial_style + tmp_colors
    if case_too_many_classes_to_be_highlighted:
        warning_messages += (
            f" \n❌Number of classes selected = {len(classes_to_be_highlighted)}" 
            f"exceeds available colors = {max_colors}."
            " Please select fewer classes for highlighting.")
        classes_to_be_highlighted = classes_to_be_highlighted[0:8]
    
    case_generate_clustnet = (btn == "btn-run-clustnet" and spec_id_selection)
    case_generate_clustnet_fails_because_no_selection = (btn == "btn-run-clustnet" and not spec_id_selection)
    
    if case_generate_clustnet:
        elements, styles, n_omitted_edges = clustnet.generate_cluster_node_link_diagram_cythonized(
            GLOBAL_DATA.tsne_coordinates_table, spec_id_selection, GLOBAL_DATA.scores_ms2deepscore, all_class_level_assignments,
            threshold, GLOBAL_DATA.sources, GLOBAL_DATA.targets, GLOBAL_DATA.values, GLOBAL_DATA.get_spectrum_iloc_list(),
            max_edges_clustnet, max_edges_per_node)
        if n_omitted_edges != int(0):
            warning_messages += (
                f"  \n❌Current settings (threshold, maximum node degree) lead to edge omission." 
                f" {n_omitted_edges} edges with lowest edge weight removed from visualization.")
    if case_generate_clustnet_fails_because_no_selection:
        warning_messages += (f"\n❌ No nodes selected, no edges can be rendered.")
    
    case_generate_egonet = (btn == "btn-run-egonet" and spec_id_selection and len(spec_id_selection) == 1)
    case_generate_egonet_fails_because_no_selection = (btn == "btn-run-egonet" and not spec_id_selection)
    case_generate_egonet_fails_because_multiselection = (
        btn == "btn-run-egonet" and spec_id_selection and len(spec_id_selection)>1)
    if case_generate_egonet:
        elements, styles, n_omitted_edges = egonet.generate_egonet_cythonized(
            spec_id_selection, GLOBAL_DATA.sources, GLOBAL_DATA.targets, GLOBAL_DATA.values, GLOBAL_DATA.tsne_coordinates_table, 
            threshold, expand_level)
        if n_omitted_edges != int(0):
            warning_messages += (
                f"  \n❌Current settings (threshold, maximum node degree, hop distance) lead to edge omission."
                f"{n_omitted_edges} edges removed from visualization. These either exceeded maximum node degrees "
                "in branching tree or were low similarity edges removed to avoid exceeding maximum edge numbers.")
    if case_generate_egonet_fails_because_no_selection:
        warning_messages += (f"  \n❌ No node selected, no edges can be shown.")
    if case_generate_egonet_fails_because_multiselection:
        warning_messages += (
            f"  \n❌More than one node selected. Select single spectrum as egonode.")

    if btn == 'btn-run-degree':
        styles, legend_plot = degree_map.generate_degree_colored_elements(
            GLOBAL_DATA.sources, GLOBAL_DATA.targets, GLOBAL_DATA.values, threshold)
        if styles and legend_plot:
            styles = GLOBAL_DATA.initial_style + styles
            legend_panel = [dcc.Graph(id = 'legend', figure = legend_plot, style={"height":"8vh", })]
        else:
            styles = GLOBAL_DATA.initial_style
            warning_messages += (f"  \n❌ Threshold too stringent. All node degrees are zero.")
    

    case_change_node_selection_but_keep_style = (btn == 'spectrum-iloc-focus-dropdown' and spec_id_selection)
    if case_change_node_selection_but_keep_style:
        styles = GLOBAL_DATA.initial_style + previous_stylesheet 
        elements = previous_elements
    
    case_highlight_selected_classes_unless_other_view_requested = (
        btn == 'dropdown_classes_to_be_highlighted' 
        # 
        and not case_generate_clustnet
        and not case_generate_egonet
        or (not btn in ("btn-run-degree"))
        and classes_to_be_highlighted 
        and not spec_id_selection
    )
    if case_highlight_selected_classes_unless_other_view_requested:
        tmp_colors = [{
            'selector' : f".{str(elem)}", 'style' : {"background-color" : COLORS[idx]}} 
            for idx, elem in enumerate(classes_to_be_highlighted)]
        styles = GLOBAL_DATA.initial_style + tmp_colors
    
    return elements, styles, zoom_level, pan_location, legend_panel, warning_messages
    
####################################################################################################################
# Set focus ids and open focus menu
@app.callback(
    Output('spectrum-iloc-focus-dropdown', 'value'),
    Output('spectrum-iloc-focus-dropdown', 'options'),
    Input('cytoscape-tsne', 'selectedNodeData'),
    Input('session_data_update_trigger', 'data'),
    State("spectrum-iloc-focus-dropdown", "options"),
    #prevent_initial_call=True
)
def displaySelectedNodeData(
    selection_data, 
    _empty_trigger, 
    old_options):

    btn = ctx.triggered_id
    if btn == 'session_data_update_trigger':
        new_options = GLOBAL_DATA.get_spectrum_iloc_list()
        new_focus_id_values = [] # always start wit empty set
        return new_focus_id_values, new_options
    
    if selection_data:
        focus_ids = []
        for elem in selection_data:
            focus_ids.append(int(elem["id"]))
        return focus_ids, old_options
    else:
        return [], old_options

@app.callback(
    Output("offcanvas-focus", "is_open"),
    Input("btn-open-focus", "n_clicks"),
    State("offcanvas-focus", "is_open"),
    prevent_initial_call=True
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

####################################################################################################################
# open settings panel
@app.callback(
    Output("offcanvas-settings", "is_open"),
    Input("btn-open-settings", "n_clicks"),
    State("offcanvas-settings", "is_open"),
    prevent_initial_call=True
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

####################################################################################################################
# details panel triggers
@app.callback(
    Output('details_panel', 'children'),
    Output('warning-messages-panel2', 'children'),
    Input('btn_push_fragmap', 'n_clicks'), 
    Input('btn_push_meta', 'n_clicks'),
    Input('btn-push-augmap', 'n_clicks'), 
    Input('btn_push_spectrum', 'n_clicks'),
    State('spectrum-iloc-focus-dropdown', 'value'),
    State("edge_threshold", "data"),
    State('colorblind_friendly_boolean_store', 'data'),
    State('top_k_fragments', 'value'),
    prevent_initial_call=True
)
def details_trigger(
    btn_fragmap_n_clicks, 
    btn_meta_n_clicks, 
    btn_augmap_n_clicks, 
    btn_spectrum_n_clicks, 
    selection_data, 
    threshold,
    colorblind_boolean, top_k_fragmap):
    """ Wrapper function that calls fragmap generation modules. """
    warning_message = ""
    btn = ctx.triggered_id
    max_number_augmap = 500
    max_number_fragmap = 40
    max_number_specplot = 25
    if btn == "btn_push_fragmap" and selection_data and len(selection_data) >= 2 and len(selection_data) <= max_number_fragmap:
        panel = fragmap.generate_fragmap_panel(selection_data, GLOBAL_DATA.spectra, top_k_fragmap)
    elif btn == "btn_push_meta" and selection_data:
        tmpdf = GLOBAL_DATA.metadata_table.iloc[selection_data]
        #tmpdf = GLOBAL_DATA.tsne_coordinates_table.iloc[selection_data]
        panel = dash_table.DataTable(
            id="table",
            columns=[{"name": i, "id": i} for i in tmpdf.columns],
            data=tmpdf.to_dict("records"),
            style_cell=dict(textAlign="left"),
            style_header=dict(backgroundColor="#8B008B", color="white", border = '1px solid black' ),
            #style_data=dict(backgroundColor="white"),
            sort_action="native",
            page_size=10,
            style_table={"overflowX": "auto"},)
    elif btn == "btn-push-augmap" and selection_data and len(selection_data) >= 2 and len(selection_data) <= max_number_augmap:
        panel = augmap.generate_augmap_panel(
            selection_data, GLOBAL_DATA.scores_ms2deepscore, GLOBAL_DATA.scores_modified_cosine , GLOBAL_DATA.scores_spec2vec, 
            threshold, colorblind_boolean)
    elif btn == "btn_push_spectrum" and selection_data and len(selection_data) <=max_number_specplot:
        if len(selection_data) == 1:
            panel = dcc.Graph(
                id="specplot", 
                figure=specplot.generate_single_spectrum_plot(GLOBAL_DATA.spectra[selection_data[0]]))
        if len(selection_data) == 2:
            panel = dcc.Graph(
                id="specplot", 
                figure=specplot.generate_mirror_plot(
                    GLOBAL_DATA.spectra[selection_data[0]], GLOBAL_DATA.spectra[selection_data[1]]))
        if len(selection_data) > 2:
            spectra = [GLOBAL_DATA.spectra[i] for i in selection_data]
            panel = specplot.generate_multiple_spectra_figure_div_list(spectra)
    else:
        panel = []
        warning_message = (
            "  \n❌ Insufficient or too many spectra selected for requested details view.")
    return panel, warning_message


####################################################################################################################
# update edge threshold
@app.callback(
    Output("edge_threshold", "data"),
    Output("edge_threshold_input_id", "placeholder"),
    Input('edge_threshold_input_id', 'n_submit'),
    Input("edge_threshold_input_id", "value"),
    prevent_initial_call=True
)

def update_threshold_trigger_handler(n_submit, new_threshold):
    new_threshold, new_placeholder=data_transfer.update_threshold(new_threshold)
    return new_threshold, new_placeholder

####################################################################################################################
# mouseover node information display
# dbc.Col([html.P("Hover Info: placeholder for cyto hover information")], id = "hover-info-panel")
@app.callback(
    Output("hover-info-panel", 'children'),
    Input('cytoscape-tsne', 'mouseoverNodeData'),
    State('select_class_level_dropdown', 'value'),
    prevent_initial_call=True
)
def displaymouseoverData(data, selected_class_level):
    """ Callback Function renders class table information and id for hovered over node in text panel."""
    if data:
        spec_id = data['id']
        node_class_info = GLOBAL_DATA.class_table.iloc[int(spec_id)].to_dict()

        selected_class_info = GLOBAL_DATA.class_table.iloc[int(spec_id)].to_dict()[selected_class_level]
        del node_class_info[selected_class_level]
        class_string_list = []
        for key in node_class_info:
            class_string_list.append(f"{key} : {node_class_info[key]} ")
        class_string = ", ".join(class_string_list)
        information_string_list = [
            html.B(f"spectrum_id: {spec_id},"), 
            html.Br(),
            html.B(f"Selected ontology = {selected_class_level}"),
            html.Br(),
            html.B(f"Predicted ontology value: {selected_class_info}"),
            html.Br(),
            f"Other classification information: {class_string}"
            ]
        return information_string_list

#####################################################################################################################
# expand level control setting
@app.callback(
    Output("expand_level", "data"),
    Output("expand_level_input", "placeholder"),
    Input('expand_level_input', 'n_submit'),
    Input("expand_level_input", "value"))

def expand_trigger_handler(n_submit, new_expand_level):
    new_expand_level, new_placeholder=data_transfer.update_expand_level(
        new_expand_level)
    return new_expand_level, new_placeholder

# expand level control setting
@app.callback(
    Output("maximum-number-of-nodes", "data"),
    Output("input-maximum-number-of-nodes", "placeholder"),
    Input("input-maximum-number-of-nodes", 'n_submit'),
    Input("input-maximum-number-of-nodes", "value"))

def max_degree_trigger_handler(n_submit, new_max_degree):
    new_max_degree, new_placeholder=data_transfer.update_max_degree(new_max_degree)
    return new_max_degree, new_placeholder
"input-maximum-number-of-nodes"


#####################################################################################################################
# CLASS SELECTION UPDATE ------------------------------------------------------
@app.callback(
    Output("selected_class_level_store", "data"), 
    Output("select_class_level_dropdown", "options"), 
    Output('selected_class_level_assignments_store', "data"),
    Output('dropdown_classes_to_be_highlighted', 'options'), 
    Output('dropdown_classes_to_be_highlighted', 'value'),
    Output('node_elements_store', 'data'),
    Output("select_class_level_dropdown", "value"),
    Input("select_class_level_dropdown", "value"),
    Input('session_data_update_trigger', 'data') # empty, global data structures updated which require selected_class_level_assignments_store to update
)
def class_update_trigger_handler(selected_class : str, _ : None):
    """ Wrapper Function that construct class dcc.store data. """
    class_levels = list(GLOBAL_DATA.class_dict.keys())
    btn = ctx.triggered_id
    
    # Update selected_class if the trigger for class updating is a new session data.
    if btn == 'session_data_update_trigger':
        selected_class = list(GLOBAL_DATA.class_dict.keys())[0]
    selected_class_level_assignments = GLOBAL_DATA.class_dict[selected_class]
    unique_assignments = list(np.unique(selected_class_level_assignments))
    node_elements = other_utils.initialize_cytoscape_graph_elements(
        GLOBAL_DATA.tsne_coordinates_table, selected_class_level_assignments, GLOBAL_DATA.highlight_table['highlight_bool'].to_list())
    return selected_class, class_levels, selected_class_level_assignments, unique_assignments, [], node_elements, selected_class

@app.callback(
    Output("edge-histogram-figure", "figure"),
    Input("edge_threshold", "data"),
    Input('session_data_update_trigger', 'data')
)
def update_histogram(threshold : float, _ : None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        y=GLOBAL_DATA.values, opacity = 0.6, 
        ybins=dict(start=0, end=1,size=0.05), marker_color = "grey")) #, histnorm="percent"
    fig.add_hline(y=threshold, line_dash = 'dash', line_color = 'magenta', line_width = 5)
    fig.update_traces(marker_line_width=1,marker_line_color="black")
    fig.update_layout(
        barmode='group', bargap=0, bargroupgap=0.0, yaxis = {'range' :[-0.01,1.01]})
    fig.update_layout(
        template="simple_white", 
        title= "Edge Weight Distibution with Threshold",
        xaxis_title="Count", yaxis_title="Edge Weight Bins",
        margin = {"autoexpand":True, "b" : 10, "l":10, "r":10, "t":40})
    return fig



@app.callback(Output('session_data_update_trigger', 'data'),
            Input('upload-data', 'value'),
            Input('scaler_id', 'value'))
def update_session_data(filename : str, scaler : Union[int, float]) -> dict:    
    '''
    Session data update handling function. Triggers when a new data path is provided or a new coordinate scaling value
    is provided. Function has a mock output for reactivity; session_data_update_trigger does not actually contain any
    session data. This is to avoid having to json serialize potentially large specXplore objects.

    Parameters:
    -----------
        filename: a string with absolute or relative filepath to data.
        scaler: an integer scaler between 0.01 and 9999. This setting is used to rescale x and y coordinates from
        the standard normal range to a larger value range for less overlap of nodes.

    Modifies:
    ---------
        The global variable GLOBAL_DATA is modified by this function.
    
    Returns: 
    --------
        An empty dictionary. 
    
    Details:
    --------
    Two cases are recognized:
    1) If the scaler triggers the function: update the specXplore object without reloading the data
    2) If the upload triggers, update the specXplore by importing data and using scaler_id value
    '''
    # 
    trigger_component = ctx.triggered_id

    case_upload_new_session_data = (
        trigger_component == "upload-data" 
        and filename is not None 
        and os.path.isfile(filename))
    case_update_scaling = (
        trigger_component == "scaler_id" 
        and scaler is not None
        and isinstance(scaler, (int, float))
        and scaler >= 0.01 
        and scaler <= 9999
    )
    if case_update_scaling or case_upload_new_session_data:
        global GLOBAL_DATA
    if case_update_scaling:
        GLOBAL_DATA.scale_coordinate_system(scaler)
        print("Coordinate system scaled.")
    elif case_upload_new_session_data:
        # check for valid and existing file
        # load file
        with open(filename, 'rb') as handle:
            specxplore_object = pickle.load(handle) 
        # assess compatibility of output
        if isinstance(specxplore_object, specxplore.datastructures.specxplore_session_data):
            GLOBAL_DATA = specxplore_object
            GLOBAL_DATA.scale_coordinate_system(scaler)
            print("Session data updated.")
        else: 
            print("Input file wrong format. No data update.")
    else:
        ...
    return {"None": None}


@app.callback(
    Output('colorblind_friendly_boolean_store', 'data'),
    Input('augmap_colorblind_friendly_switch', 'on')
)
def update_output(on_off_state):
    return on_off_state

if __name__ == "__main__":    
    app.run(debug=True, port="8999")