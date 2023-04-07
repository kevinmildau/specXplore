# Main specXplore prototype
from dash import dcc, html, ctx, dash_table, Dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from specxplore import egonet
from specxplore import augmap
from specxplore import clustnet
from specxplore import fragmap
from specxplore import data_transfer
from specxplore import specxplore_data_cython
from specxplore import spectrum_plot
from specxplore import degree_visualization
from specxplore import other_utils
from specxplore.clustnet import SELECTED_NODES_STYLE, GENERAL_STYLE, SELECTION_STYLE
import pickle
import plotly.graph_objects as go
import os
import numpy as np
import pandas as pd # tmp import
from specxplore.specxplore_data import specxplore_data, Spectrum
from specxplore.constants import COLORS

# Load specxplore data object from file
if False:
    specxplore_input_file = os.path.join("data_import_testing", "results", "phophe_specxplore.pickle")
    with open(specxplore_input_file, 'rb') as handle:
        GLOBAL_DATA = pickle.load(handle) 
#if True:
#    specxplore_input_file = 'data_and_output/wheat_data/wheat_data_specxplore_v2.pickle'
#    #specxplore_input_file = os.path.join("data_and_output", "npl_out", "npl_specxplore.pickle")
#    with open(specxplore_input_file, 'rb') as handle:
#        GLOBAL_DATA = pickle.load(handle) 

if False:
    specxplore_input_file = 'data_and_output/test_data/test_case_specxplore2.pickle'
    #specxplore_input_file = os.path.join("data_and_output", "npl_out", "npl_specxplore.pickle")
    with open(specxplore_input_file, 'rb') as handle:
        GLOBAL_DATA = pickle.load(handle) 

if True:
    specxplore_input_file = 'data_and_output/urine_data/urine_data_specxplore.pickle'
    #specxplore_input_file = os.path.join("data_and_output", "npl_out", "npl_specxplore.pickle")
    with open(specxplore_input_file, 'rb') as handle:
        GLOBAL_DATA = pickle.load(handle) 

# Unpack specXplore input object
GLOBAL_DATA.class_table["is_standard"] = pd.Series(GLOBAL_DATA.is_standard, dtype = str) # tmp modification
CLASS_DICT = {elem : list(GLOBAL_DATA.class_table[elem]) for elem in GLOBAL_DATA.class_table.columns} 
AVAILABLE_CLASSES = list(CLASS_DICT.keys())
selected_class_data = CLASS_DICT[AVAILABLE_CLASSES[0]]


SM_MS2DEEPSCORE = GLOBAL_DATA.ms2deepscore_sim
SM_MODIFIED_COSINE = GLOBAL_DATA.cosine_sim 
SM_SPEC2VEC = GLOBAL_DATA.spec2vec_sim
TSNE_DF = GLOBAL_DATA.tsne_df
scaler = 1200
TSNE_DF["x"] = other_utils.scale_array_to_minus1_plus1(TSNE_DF["x"].to_numpy()) * scaler
TSNE_DF["y"] = other_utils.scale_array_to_minus1_plus1(TSNE_DF["y"].to_numpy()) * scaler
ALL_SPEC_IDS = GLOBAL_DATA.specxplore_id
ALL_SPECTRA = GLOBAL_DATA.spectra
SOURCE = GLOBAL_DATA.sources
TARGET = GLOBAL_DATA.targets
VALUE = GLOBAL_DATA.values
MZ = GLOBAL_DATA.mz

def initialize_cytoscape_graph_elements(tsne_df, selected_class_data, is_standard):
    n_nodes = tsne_df.shape[0]
    nodes = [{}] * n_nodes 
    for i in range(0, n_nodes):
        if is_standard[i] == True:
            standard_entry = " is_standard"
        else:
            standard_entry = ""
        nodes[i] = {
            'data':{'id':str(i)},
            'classes': str(selected_class_data[i]) + standard_entry, #.replace('_',''), #  color_class[i],
            'position':{'x':TSNE_DF["x"].iloc[i], 'y':-TSNE_DF["y"].iloc[i]}        
        }
    return nodes

INITIAL_NODE_ELEMENTS = initialize_cytoscape_graph_elements(TSNE_DF, selected_class_data, GLOBAL_DATA.is_standard)
INITIAL_STYLE = SELECTED_NODES_STYLE + GENERAL_STYLE + SELECTION_STYLE

########################################################################################################################
# All settings panel
upload_element = html.Div([dcc.Input(id="upload-data", type="text", placeholder=None, debounce=True)])

########################################################################################################################
# All settings panel
settings_panel = dbc.Offcanvas([
    html.P("SpecXplore defaults and limits can be modified here."),
    html.B("Set Edge Threshold:"),
    html.P("A value between 0 and 1 (excluded)"),
    dcc.Input(
        id="edge_threshold_input_id", type="number", 
        debounce=True, placeholder="Threshold 0 < thr < 1, def. 0.9", 
        style={"width" : "100%"}),
    html.Div([dcc.Graph(
        id = "edge-histogram-figure", figure = [],  style={"width":"100%","height":"30vh"})]),
    html.B("Set Visal Scale:"),
    html.P("A value between 0 exclded and 2000. This value is used to expand the t-SNE layout to be larger or smaller."),
    dcc.Input(
        id="scaler_id", type="number", 
        debounce=True, value=100, placeholder = "100",
        style={"width" : "100%"}),
    html.B("Set expand level:"),
    html.P("A value between 1 and 5. Controls connection branching in EgoNet."),
    dcc.Input( id="expand_level_input", type="number", 
        debounce=True, 
        placeholder="Value between 1 < exp.lvl. < 5, def. 1", 
        style={"width" : "100%"}),
    html.B("Select Class Level:"),
    dcc.Dropdown(
        id='select_class_level_dropdown' , multi=False, 
        clearable=False, options=AVAILABLE_CLASSES, 
        value=AVAILABLE_CLASSES[5]),
    html.B("Filter by Class(es):"),
    dcc.Dropdown(
        id='classes_to_be_highlighted_dropdown' , multi=True, clearable=False, options=[],
        value=[]),
    ],
    id="offcanvas-settings",
    title="Settings Panel",
    is_open=False,)

########################################################################################################################
# Focus selection panel
selection_focus_panel = dbc.Offcanvas([
    html.P((
        "All spectrum ids selected for focused analysis in network views."
        "\n Selection can be modified here.")),
    dcc.Dropdown(id='specid-focus-dropdown', multi=True, 
            style={'width': '90%', 'font-size': "100%"}, 
            options=ALL_SPEC_IDS)],
    id="offcanvas-focus",
    placement="end",
    title="Selection Panel",
    is_open=False)
########################################################################################################################
# Button array on left side of plot
height_small = "9vh" # round down of 80vh / 7 components
height_main = "9vh" # round down of 80vh / 7 components + 80%7 /2 
button_style_logo = {'width': '25%', 'height': height_small, "fontSize": "20px", "textAlign": "center", }
button_style_text = {'width': '75%', 'height': height_small, "fontSize": "10px", "textAlign": "center", }
control_button_group = [
    dbc.ButtonGroup(
        [
        dbc.Button('⚙', id="btn-open-settings", style = 
                   {'width': '100%', 'height': height_main, "fontSize": "25px"}),
        dbc.Button('View Selection', id="btn-open-focus", style = 
                   {'width': '100%', 'height': height_main, "fontSize": "15px"}),
        dbc.ButtonGroup([
            dbc.Button('👁', id='btn-nclicks0-settings', style = button_style_logo),
            dbc.Button('Show Node Degree ', id='btn-run-degree', n_clicks=0, style = button_style_text),
            ]),  
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-nclicks1-settings', style = button_style_logo),
            dbc.Button('EgoNet ', id='btn-run-egonet', n_clicks=0, style = button_style_text),
            ]),  
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-nclicks2-settings', style = button_style_logo),
            dbc.Button('Show Edges for Selection', id="btn-run-clustnet", n_clicks=0, style = button_style_text),  
        ]),
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-nclicks3-settings', style = button_style_logo),  
            dbc.Button('Augmap', id="btn-push-augmap", n_clicks=0, style = button_style_text)
        ]),
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-nclicks4-settings', style = button_style_logo),  
            dbc.Button('Spectrum Plot(s)', id='btn_push_spectrum', n_clicks=0, style = button_style_text)
        ]),
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-fragmap-settings', style = button_style_logo),  
            dbc.Button('Fragmap', id="btn_push_fragmap", n_clicks=0, style = button_style_text)
        ]),
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-meta-settings', style = button_style_logo),  
            dbc.Button('Metadata Table', id="btn_push_meta", n_clicks=0, style = button_style_text)
        ]),
        ],
        vertical=True,
    )]

########################################################################################################################
# good themes: VAPOR, UNITED ; see more:  https://bootswatch.com/
app=Dash(external_stylesheets=[dbc.themes.UNITED])
app.layout=html.Div([
    # ROW 1: Title & hover response
    upload_element,
    dbc.Row(
        [
        dbc.Col([html.H1([html.B("specXplore prototype")], style={"margin-bottom": "0em"})], width=6),
        dbc.Col([html.Tbody("Hover Info: placeholder for cyto hover information")], id = "hover-info-panel",
                style={"font-size" : "10pt"})
        ]
    ),
    html.Br(),
    dbc.Row(
        [
        dbc.Col(control_button_group, width = 2),
        dbc.Col([cyto.Cytoscape( 
            id='cytoscape-tsne', elements=INITIAL_NODE_ELEMENTS, 
            stylesheet = INITIAL_STYLE,
            layout={'name':'preset', 'animate':False, 'fit': False}, boxSelectionEnabled=True,
            style={'width':'100%', 'height':'80vh', "border":"1px grey solid", "bg":"#feeff4", 'minZoom':0.1, 'maxZoom':2})], 
            width=10) 
        ], 
        style={"margin": "0px"}, className="g-0"),
    #html.Br(),
    #dbc.Button("Open Focus", id="btn-open-focus", n_clicks=0),
    settings_panel,
    selection_focus_panel,
    dcc.Store(id='specxplore_object', data = None),
    dcc.Store(id="edge_threshold", data=0.9),
    dcc.Store(id="expand_level", data=int(1)),
    dcc.Store(id="selected_class_level_store", data=AVAILABLE_CLASSES[0]),
    dcc.Store(id='selected_class_level_assignments_store', data=CLASS_DICT[AVAILABLE_CLASSES[0]]),
    dcc.Store(id='node_elements_store', data = INITIAL_NODE_ELEMENTS),
    #html.Br(),
    dbc.Row(
        [
        dbc.Col(
            [
            dcc.Markdown(id="warning-messages-panel1", style={"font-size" : "8pt", "border":"1px grey solid"}),
            dcc.Markdown(id="warning-messages-panel2", style={"font-size" : "8pt", "border":"1px grey solid"})
            ], width=4), 
        dbc.Col([html.Div(id="legend_panel", style={"width":"100%", "border":"1px grey solid"})], width=8)
        ], className="g-0"), 
    dbc.Row(
        [
        dbc.Col([html.Div(id="details_panel", style={"width":"100%", "height":"80vh", "border":"1px grey solid"})], width=12)
        ], className="g-0"),
    ], 
    style={"width" : "100%"},
    )

########################################################################################################################
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
    Input('specid-focus-dropdown', 'value'), 
    Input('classes_to_be_highlighted_dropdown', 'value'),
    Input('selected_class_level_assignments_store', "data"),
    Input('node_elements_store', 'data'),
    Input('specxplore_object', 'data'),
    State("edge_threshold", "data"),
    State("expand_level", "data"),
    State('cytoscape-tsne', 'zoom'),
    State('cytoscape-tsne', 'pan'),
    State('cytoscape-tsne', 'elements'),   # <-- for persistency
    State('cytoscape-tsne', 'stylesheet'), # <-- for persistency
    prevent_initial_call=True)
def cytoscape_trigger(
    n_clicks1, 
    n_clicks2,
    n_clicks3,
    spec_id_selection, 
    classes_to_be_highlighted,
    all_class_level_assignments, 
    node_elements_from_store,
    none_data_trigger, 
    threshold, 
    expand_level, 
    zoom_level, 
    pan_location,
    previous_elements,
    previous_stylesheet):

    btn = ctx.triggered_id
    max_colors = 8
    max_edges_clustnet = 2500
    max_edges_egonet = 2500
    legend_panel = [] # initialize to empty list. Only get filled if degree node selected
    warning_messages = "" # initialize to empty string. Only gets expanded if warnings necessary.
    styles = INITIAL_STYLE

    elements = node_elements_from_store

    if (not btn in ("btn-run-egonet", "btn-run-clustnet") 
            and classes_to_be_highlighted 
            and len(classes_to_be_highlighted) > max_colors):
        # filter classes to be highlighted to be less than or equal to max colors in length. 
        warning_messages += (
            f"  \n❌Number of classes selected = {len(classes_to_be_highlighted)} exceeds available colors = {max_colors}."
            " Selection truncated to first 8 classes in selection dropdown.")
        classes_to_be_highlighted = classes_to_be_highlighted[0:8]
    
    if classes_to_be_highlighted and btn == 'classes_to_be_highlighted_dropdown':
        # define style color update if trigger is class highlight dropdown
        tmp_colors = [{
            'selector' : f".{str(elem)}", 'style' : {"background-color" : COLORS[idx]}} 
            for idx, elem in enumerate(classes_to_be_highlighted)]
        styles = INITIAL_STYLE + tmp_colors
    
    if btn == "btn-run-clustnet":
        if spec_id_selection:
            elements, styles, n_omitted_edges = clustnet.generate_cluster_node_link_diagram_cythonized(
                TSNE_DF, spec_id_selection, GLOBAL_DATA.ms2deepscore_sim, all_class_level_assignments,
                threshold, SOURCE, TARGET, VALUE, MZ, GLOBAL_DATA.is_standard,
                max_edges_clustnet)
            if spec_id_selection and n_omitted_edges != int(0):
                warning_messages += (
                    f"  \n❌Threshold too liberal and leads to number of edges exceeding allowed maximum." 
                    f" {n_omitted_edges} edges with lowest edge weight removed from visualization.")
        else:
            warning_messages += (
                f"\n❌ No nodes selected, no edges can be rendered."
            )

    if btn == "btn-run-egonet":
        if spec_id_selection:
            elements, styles, n_omitted_edges = egonet.generate_egonet_cythonized(
                spec_id_selection, SOURCE, TARGET, VALUE, TSNE_DF, MZ, 
                threshold, expand_level)
            if n_omitted_edges != int(0):
                warning_messages += (
                    f"  \n❌Threshold too liberal and leads to number of edges exceeding allowed maximum. "
                    f"{n_omitted_edges} edges with lowest edge weight removed from visualization.")
        else:
            warning_messages += (f"  \n❌ No nodes selected, no edges can be shown.")
        if (spec_id_selection and len(spec_id_selection) > 1):
            warning_messages += (
                f"  \n❌More than one node selected. Selecting spectrum {spec_id_selection[0]} as ego node.")

    if btn == 'btn-run-degree':
        styles, legend_plot = degree_visualization.generate_degree_colored_elements(SOURCE, TARGET, VALUE, threshold)
        if styles and legend_plot:
            styles = INITIAL_STYLE + styles
            legend_panel = [dcc.Graph(id = 'legend', figure = legend_plot, style={"height":"8vh", })]
        else:
            styles = INITIAL_STYLE
            warning_messages += (f"  \n❌ Threshold too stringent. All node degrees are zero.")
    if (btn == 'specid-focus-dropdown' and spec_id_selection):
        styles = INITIAL_STYLE + previous_stylesheet 
        elements = previous_elements
    
    if (not btn in ("btn-run-egonet", "btn-run-clustnet") and classes_to_be_highlighted and not spec_id_selection):
        tmp_colors = [{
            'selector' : f".{str(elem)}", 'style' : {"background-color" : COLORS[idx]}} 
            for idx, elem in enumerate(classes_to_be_highlighted)]
        styles = INITIAL_STYLE + tmp_colors

    #warning_messages = f"❌Warnings Panel: this is where warnings live. What have you done (ಠ_ಠ)! 1234567890 (ಥ_ಥ)."

    return elements, styles, zoom_level, pan_location, legend_panel, warning_messages
    
########################################################################################################################
# Set focus ids and open focus menu
@app.callback(
    Output('specid-focus-dropdown', 'value'),
    Input('cytoscape-tsne', 'selectedNodeData'),
    prevent_initial_call=True
)
def displaySelectedNodeData(data):
    if data:
        focus_ids = []
        for elem in data:
            focus_ids.append(int(elem["id"]))
        return focus_ids

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

########################################################################################################################
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

########################################################################################################################
# details panel triggers
@app.callback(
    Output('details_panel', 'children'),
    Output('warning-messages-panel2', 'children'),
    Input('btn_push_fragmap', 'n_clicks'), 
    Input('btn_push_meta', 'n_clicks'),
    Input('btn-push-augmap', 'n_clicks'), 
    Input('btn_push_spectrum', 'n_clicks'),
    State('specid-focus-dropdown', 'value'),
    State("edge_threshold", "data"),
    prevent_initial_call=True
)
def details_trigger(
    btn_fragmap_n_clicks, btn_meta_n_clicks, btn_augmap_n_clicks, btn_spectrum_n_clicks, selection_data, threshold):
    """ Wrapper function that calls fragmap generation modules. """
    warning_message = ""
    btn = ctx.triggered_id
    max_number_augmap = 200
    max_number_fragmap = 25
    max_number_specplot = 25
    if btn == "btn_push_fragmap" and selection_data and len(selection_data) >= 2 and len(selection_data) <= max_number_fragmap:
        panel = fragmap.generate_fragmap_panel(selection_data, ALL_SPECTRA)
    elif btn == "btn_push_meta" and selection_data:
        
        tmpdf = TSNE_DF.iloc[selection_data]
        panel = dash_table.DataTable(
            id="table",
            columns=[{"name": i, "id": i} for i in tmpdf.columns],
            data=tmpdf.to_dict("records"),
            style_cell=dict(textAlign="left"),
            style_header=dict(backgroundColor="orangered"),
            #style_data=dict(backgroundColor="white"),
            sort_action="native",
            page_size=10,
            style_table={"overflowX": "auto"},)
    elif btn == "btn-push-augmap" and selection_data and len(selection_data) >= 2 and len(selection_data) <= max_number_augmap:
        panel = augmap.generate_augmap_panel(
            selection_data, GLOBAL_DATA.ms2deepscore_sim, GLOBAL_DATA.cosine_sim , GLOBAL_DATA.spec2vec_sim, 
            threshold)
    elif btn == "btn_push_spectrum" and selection_data and len(selection_data) <=max_number_specplot:
        if len(selection_data) == 1:
            panel = dcc.Graph(
                id="specplot", 
                figure=spectrum_plot.generate_single_spectrum_plot(ALL_SPECTRA[selection_data[0]]))
        if len(selection_data) == 2:
            panel = dcc.Graph(
                id="specplot", 
                figure=spectrum_plot.generate_mirror_plot(
                    ALL_SPECTRA[selection_data[0]], ALL_SPECTRA[selection_data[1]]))
        if len(selection_data) > 2:
            spectra = [ALL_SPECTRA[i] for i in selection_data]
            panel = spectrum_plot.generate_multiple_spectra_figure_div_list(spectra)
    else:
        panel = []
        warning_message = (
            "  \n❌ Insufficient or too many spectra selected for requested details view.")
    return panel, warning_message


########################################################################################################################
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

########################################################################################################################
# mouseover node information display
# dbc.Col([html.P("Hover Info: placeholder for cyto hover information")], id = "hover-info-panel")
@app.callback(
    Output("hover-info-panel", 'children'),
    Input('cytoscape-tsne', 'mouseoverNodeData'),
    State('selected_class_level_assignments_store', "data"),
    prevent_initial_call=True
)
def displaymouseoverData(data, class_assignment_list):
    if data:
        spec_id = data['id']
        metdata_information = TSNE_DF.iloc[int(spec_id)].to_dict()
        information_string = (f"spectrum_id: {spec_id}, class: {class_assignment_list[int(spec_id)]} ")
        return information_string

########################################################################################################################
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

########################################################################################################################
# CLASS SELECTION UPDATE ------------------------------------------------------
@app.callback(
    Output("selected_class_level_store", "data"), 
    Output('selected_class_level_assignments_store', "data"),
    Output('classes_to_be_highlighted_dropdown', 'options'), 
    Output('classes_to_be_highlighted_dropdown', 'value'),
    Output('node_elements_store', 'data'),
    Input("select_class_level_dropdown", "value"),
    Input('specxplore_object', 'data') # empty, global data strctures updated which require selected_class_level_assignments_store to update
)
def class_update_trigger_handler(selected_class : str, _ : None):
    """ Wrapper Function that construct class dcc.store data. """
    selected_class_level_assignments = CLASS_DICT[selected_class]
    unique_assignments = list(np.unique(selected_class_level_assignments))

    node_elements = initialize_cytoscape_graph_elements(
        TSNE_DF, selected_class_level_assignments, GLOBAL_DATA.is_standard)
    
    
    return selected_class, selected_class_level_assignments, unique_assignments, [], node_elements

@app.callback(
    Output("edge-histogram-figure", "figure"),
    Input("edge_threshold", "data")
)
def update_histogram(threshold):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        y=VALUE, opacity = 0.6, 
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

@app.callback(Output('specxplore_object', 'data'),
              Input('upload-data', 'value'),
              State('scaler_id', 'value'))
def update_output(filename, scaler):
    print(scaler, type(scaler))
    if filename is not None:
        # check for valid and existing file
        # load file
        with open(filename, 'rb') as handle:
            specxplore_object = pickle.load(handle) 
        # assess compatibility of output
        assert type(specxplore_object) == specxplore_data
        # Unpack specXplore input object
        global ALL_SPEC_IDS
        global ALL_SPECTRA
        global SOURCE
        global TARGET
        global VALUE
        global MZ
        global INITIAL_NODE_ELEMENTS
        global INITIAL_STYLE
        global TSNE_DF
        global SM_MODIFIED_COSINE
        global SM_MS2DEEPSCORE
        global SM_SPEC2VEC
        global AVAILABLE_CLASSES
        global CLASS_DICT
        GLOBAL_DATA = specxplore_object
        GLOBAL_DATA.class_table["is_standard"] = pd.Series(GLOBAL_DATA.is_standard, dtype = str) # tmp modification
        CLASS_DICT = {elem : list(GLOBAL_DATA.class_table[elem]) for elem in GLOBAL_DATA.class_table.columns} 
        AVAILABLE_CLASSES = list(CLASS_DICT.keys())
        selected_class_data = CLASS_DICT[AVAILABLE_CLASSES[0]]
        SM_MS2DEEPSCORE = GLOBAL_DATA.ms2deepscore_sim
        SM_MODIFIED_COSINE = GLOBAL_DATA.cosine_sim 
        SM_SPEC2VEC = GLOBAL_DATA.spec2vec_sim
        TSNE_DF = GLOBAL_DATA.tsne_df
        TSNE_DF["x"] = other_utils.scale_array_to_minus1_plus1(TSNE_DF["x"].to_numpy()) * scaler
        TSNE_DF["y"] = other_utils.scale_array_to_minus1_plus1(TSNE_DF["y"].to_numpy()) * scaler
        ALL_SPEC_IDS = GLOBAL_DATA.specxplore_id
        ALL_SPECTRA = GLOBAL_DATA.spectra
        SOURCE = GLOBAL_DATA.sources
        TARGET = GLOBAL_DATA.targets
        VALUE = GLOBAL_DATA.values
        MZ = GLOBAL_DATA.mz
        INITIAL_NODE_ELEMENTS = initialize_cytoscape_graph_elements(TSNE_DF, selected_class_data, GLOBAL_DATA.is_standard)
        INITIAL_STYLE = SELECTED_NODES_STYLE + GENERAL_STYLE + SELECTION_STYLE
        print("specXplore data sccessfully updated")
        return {"None": None}

    



if __name__ == '__main__':
    app.run_server(debug=True, port="8999")


                                                                               
