# Main specXplore prototype
from dash import dcc, html, ctx, dash_table, Dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
from specxplore import egonet, augmap, tsne_plotting, clustnet, fragmap, data_transfer, specxplore_data_cython, specxplore_data
import pickle
import plotly.graph_objects as go
import os

# Load specxplore data object from file
specxplore_input_file = os.path.join("data_import_testing", "results", "phophe_specxplore.pickle")
with open(specxplore_input_file, 'rb') as handle:
    GLOBAL_DATA = pickle.load(handle) 




# Define global variables
global CLASS_DICT # Dictionary with key for each possible classification scheme, and list with class assignments
global AVAILABLE_CLASSES # List of class strings corresponding to keys in class_dict
global SM_MS2DEEPSCORE # ndarray - pairwise similarity matrix
global SM_MODIFIED_COSINE # ndarray - pairwise similarity matrix
global SM_SPEC2VEC # ndarray - pairwise similarity matrix
global TSNE_DF # pandas.DataFrame with x, y, is_standard, and id columns
global ALL_SPEC_IDS
global ALL_SPECTRA
global MZ

tmp = GLOBAL_DATA.class_table
CLASS_DICT = {elem : list(tmp[elem]) for elem in tmp.columns} 
AVAILABLE_CLASSES = list(CLASS_DICT.keys())
print(AVAILABLE_CLASSES)

# Extract pairwise similarity matrices
SM_MS2DEEPSCORE = GLOBAL_DATA.ms2deepscore_sim
SM_MODIFIED_COSINE = GLOBAL_DATA.cosine_sim 
SM_SPEC2VEC = GLOBAL_DATA.spec2vec_sim
# Extract and expand t-sne df
TSNE_DF = GLOBAL_DATA.tsne_df
TSNE_DF["is_standard"] = GLOBAL_DATA.is_standard
TSNE_DF["id"] = GLOBAL_DATA.specxplore_id

print(TSNE_DF)

def standardize_array(array : np.ndarray):
    out = (array - np.mean(array)) / np.std(array)
    return out

def scale_array_to_minus1_plus1(array : np.ndarray):
    # Normalised [-1,1]
    out = 2.*(array - np.min(array))/np.ptp(array)-1
    return out
#TSNE_DF["x"] = standardize_array(TSNE_DF["x"].to_numpy()) * 1000
#TSNE_DF["y"] = standardize_array(TSNE_DF["y"].to_numpy()) * 1000

TSNE_DF["x"] = scale_array_to_minus1_plus1(TSNE_DF["x"].to_numpy()) * 100
TSNE_DF["y"] = scale_array_to_minus1_plus1(TSNE_DF["y"].to_numpy()) * 100


# INITIALIZE COLOR_DICT
selected_class_data=CLASS_DICT[AVAILABLE_CLASSES[0]]
# INITIALIZE GRAYSCALE COLOUR MAPPING
n_colors=len(set(selected_class_data))
colors=data_transfer.construct_grey_palette(n_colors, white_buffer=20)
init_color_dict=data_transfer.create_color_dict(colors, selected_class_data)
# INITIALIZE ALL SPEC IDS LIST? NDARRAY
ALL_SPEC_IDS = GLOBAL_DATA.specxplore_id
# INITIALIZE ALL SPECTRA LIST
ALL_SPECTRA = [Spectrum(spec.peaks.mz, max(spec.peaks.mz),idx, spec.peaks.intensities) for idx, spec in enumerate(GLOBAL_DATA.spectra)]
# CONSTRUCT SOURCE, TARGET AND VALUE ND ARRAYS
SOURCE, TARGET, VALUE = specxplore_data_cython.construct_long_format_sim_arrays(SM_MS2DEEPSCORE)
# CONSTRUCT MZ DATA LIST FOR VISUALIZATION
MZ = GLOBAL_DATA.mz



from specxplore.clustnet import SELECTED_NODES_STYLE, GENERAL_STYLE, SELECTION_STYLE

def initialize_cytoscape_graph_elements(tsne_df, selected_class_data, mz):
    n_nodes = tsne_df.shape[0]
    nodes = [{}] * n_nodes 
    for i in range(0, n_nodes):
        nodes[i] = {
                'data':{'id':str(i), 
                'label': str(str(i) + ': ' + str(mz[i]))},
                'position':{'x':TSNE_DF["x"].iloc[i], 'y':-TSNE_DF["y"].iloc[i]}, 
                'classes': selected_class_data[i]}
    return nodes

initial_node_elements = initialize_cytoscape_graph_elements(TSNE_DF, selected_class_data, MZ)
initial_style = SELECTED_NODES_STYLE + GENERAL_STYLE + SELECTION_STYLE
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
    html.B("Set expand level:"),
    html.P("A value between 1 and 5. Controls connection branching in EgoNet."),
    dcc.Input( id="expand_level_input", type="number", 
        debounce=True, 
        placeholder="Value between 1 < exp.lvl. < 5, def. 1", 
        style={"width" : "100%"}),
    html.B("Select Class Level:"),
    dcc.Dropdown(
        id='class-dropdown' , multi=False, 
        clearable=False, options=AVAILABLE_CLASSES, 
        value=AVAILABLE_CLASSES[5]),
    html.B("Filter by Class(es):"),
    dcc.Dropdown(
        id='class-filter-dropdown' , multi=True, clearable=False, options=[],
        value=[]),
    ],
    id="offcanvas-settings",
    title="Settings Panel",
    is_open=False,)
########################################################################################################################
# Main selection panel
selection_panel = dbc.Offcanvas([
    html.P((
        "All selected spectrums ids from the overview graph:"
        "\n Selection can be modified here.")),
    dcc.Dropdown(id='specid-selection-dropdown', multi=True, 
            style={'width': '90%', 'font-size': "75%"}, 
            options=ALL_SPEC_IDS)],
    id="offcanvas-selection",
    placement="bottom",
    title="Selection Panel",
    is_open=False)
########################################################################################################################
# Focus selection panel
selection_focus_panel = dbc.Offcanvas([
    html.P((
        "All spectrum ids selected for focused analysis in network views."
        "\n Selection can be modified here.")),
    dcc.Dropdown(id='specid-focus-dropdown', multi=True, 
            style={'width': '90%', 'font-size': "100%"}, 
            options=ALL_SPEC_IDS),
    html.B("Enter spec_id for spectrum plot:"),
    dcc.Dropdown(id='specid-spectrum_plot-dropdown', multi=False, 
        options=ALL_SPEC_IDS, style={'width' : '50%'}),
    html.B("Enter spec_id1 (top) and spec_id2 (bottom) for mirrorplot:"),
    dcc.Dropdown(id='specid-mirror-1-dropdown', multi=False, 
        options=ALL_SPEC_IDS,
        style={'width' : '50%'}),
    dcc.Dropdown(id='specid-mirror-2-dropdown', multi=False, 
        options=ALL_SPEC_IDS,
        style={'width' : '50%'})],
    id="offcanvas-focus",
    placement="end",
    title="Focus Selection Panel",
    is_open=False)
########################################################################################################################
# Button array on left side of plot
height_small = "11vh" # round down of 80vh / 7 components
height_main = "14vh" # round down of 80vh / 7 components + 80%7 
button_style_logo = {'width': '25%', 'height': height_small, "fontSize": "20px", "textAlign": "center", }
button_style_text = {'width': '75%', 'height': height_small, "fontSize": "10px", "textAlign": "center", }
control_button_group = [
    dbc.ButtonGroup(
        [
        dbc.Button('⚙', id="btn-open-settings", style = 
                   {'width': '100%', 'height': height_main, "fontSize": "25px"}),
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-nclicks1-settings', style = button_style_logo),
            dbc.Button('EgoNet ', id='btn-run-egonet', n_clicks=0, style = button_style_text),
            ]),  
        dbc.ButtonGroup([
            dbc.Button('⚙', id='btn-nclicks2-settings', style = button_style_logo),
            dbc.Button('ClustNet', id="btn-run-clustnet", n_clicks=0, style = button_style_text),  
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
    dbc.Row(
        [
        dbc.Col([html.H1([html.B("specXplore prototype")], style={"margin-bottom": "0em"})], width=6),
        dbc.Col([html.P("Hover Info: placeholder for cyto hover information")], id = "hover-info-panel")
        ]
    ),
    html.Br(),
    dbc.Row(
        [
        dbc.Col(control_button_group, width = 1),
        dbc.Col([cyto.Cytoscape( 
            id='cytoscape-tsne', elements=initial_node_elements, 
            stylesheet = initial_style,
            layout={'name':'preset', "fit": True}, zoom = 1, boxSelectionEnabled=True,
            style={'width':'100%', 'height':'80vh', "border":"1px grey solid", "bg":"#feeff4"})], 
            width=11),
        ], 
        style={"margin": "0px"}, className="g-0"),
    html.Br(),
    dbc.Button("Open Selection", id="btn-open-selection", n_clicks=0),
    dbc.Button("Open Focus", id="btn-open-focus", n_clicks=0),
    settings_panel,
    selection_panel,
    selection_focus_panel,
    dcc.Store(id="edge_threshold", data=0.9),
    dcc.Store(id="expand_level", data=int(1)),
    dcc.Store(id="selected-filter-classes", data = []),
    dcc.Store(id="selected_class_level", data=AVAILABLE_CLASSES[0]),
    dcc.Store(id="selected_class_data", data=CLASS_DICT[AVAILABLE_CLASSES[0]]),
    dcc.Store(id="color_dict",  data=init_color_dict),
    html.Br(),
    dbc.Row([dbc.Col([html.Div(id="details_panel", style={"width":"100%", "height":"100%", "border":"1px grey solid"})], width=12)]),
    ], 
    style={"width" : "100%"},
    )

########################################################################################################################
# cytoscape update callbacks

@app.callback(
    Output('cytoscape-tsne', 'elements'),
    Output('cytoscape-tsne', 'stylesheet'),
    Input('btn-run-egonet', 'n_clicks'),
    Input('btn-run-clustnet', 'n_clicks'),
    State('specid-focus-dropdown', 'value'), 
    State("color_dict", "data"),
    State("selected_class_data", "data"),
    State("edge_threshold", "data"),
    State("expand_level", "data"))
def cytoscape_trigger(
    n_clicks1, n_clicks2, spec_id_selection, color_dict, 
    selected_class_data, threshold, expand_level):
    btn = ctx.triggered_id
    print(spec_id_selection)

    elements = initial_node_elements
    styles = initial_style
    if btn == "btn-run-clustnet" and spec_id_selection:
        elements, styles = clustnet.generate_cluster_node_link_diagram_cythonized(
            TSNE_DF, spec_id_selection, GLOBAL_DATA.ms2deepscore_sim, selected_class_data,
            color_dict, threshold, SOURCE, TARGET, VALUE, MZ)
    elif btn == "btn-run-egonet"  and spec_id_selection:
        elements, styles = egonet.generate_egonet_cythonized(
            spec_id_selection, SOURCE, TARGET, VALUE, TSNE_DF, MZ, 
            threshold, expand_level)
    else:
        elements = initial_node_elements
    return elements, styles


########################################################################################################################
# Set focus ids and open focus menu
@app.callback(
    Output('specid-focus-dropdown', 'value'),
    [Input('cytoscape-tsne', 'selectedNodeData')])
def displaySelectedNodeData(data):
    if data:
        focus_ids = []
        for elem in data:
            focus_ids.append(int(elem["id"]))
        return focus_ids

@app.callback(
    Output("offcanvas-focus", "is_open"),
    Input("btn-open-focus", "n_clicks"),
    [State("offcanvas-focus", "is_open")],
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
    [State("offcanvas-settings", "is_open")],
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open

########################################################################################################################
# details panel triggers
@app.callback(
    Output('details_panel', 'children'),
    Input('btn_push_fragmap', 'n_clicks'), 
    Input('btn_push_meta', 'n_clicks'),
    Input('btn-push-augmap', 'n_clicks'), 
    Input('btn_push_spectrum', 'n_clicks'),
    State('specid-focus-dropdown', 'value'),
    State("edge_threshold", "data"),)
def fragmap_trigger(
    btn_fragmap_n_clicks, btn_meta_n_clicks, btn_augmap_n_clicks, btn_spectrum_n_clicks, selection_data, threshold):
    """ Wrapper function that calls fragmap generation modules. """
    btn = ctx.triggered_id

    if btn == "btn_push_fragmap" and selection_data:
        panel = fragmap.generate_fragmap_panel(selection_data, ALL_SPECTRA)
    elif btn == "btn_push_meta" and selection_data:
        ...
    elif btn == "btn-push-augmap" and selection_data:
        panel = augmap.generate_augmap_panel(
            selection_data, GLOBAL_DATA.ms2deepscore_sim, GLOBAL_DATA.cosine_sim , GLOBAL_DATA.spec2vec_sim, 
            threshold)
    elif btn == "btn_push_spectrum" and selection_data:
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
            
        ...
    else:
        panel = [html.P((
            "Augmap, Fragmap, Metadata table, and Spectrum plots are shown here"
            " once requested with valid input."))]
        
    if False:
        if selection_data:
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
            return panel
        else:
            panel = [html.H6((
                "Select focus data and press " 
                "'Show Metdata' button for data table."))]
        return panel
    return panel


########################################################################################################################
# update edge threshold
@app.callback(
    Output("edge_threshold", "data"),
    Output("edge_threshold_input_id", "placeholder"),
    Input('edge_threshold_input_id', 'n_submit'),
    Input("edge_threshold_input_id", "value"))

def update_threshold_trigger_handler(n_submit, new_threshold):
    new_threshold, new_placeholder=data_transfer.update_threshold(new_threshold)
    return new_threshold, new_placeholder

########################################################################################################################
# mouseover node information display
# dbc.Col([html.P("Hover Info: placeholder for cyto hover information")], id = "hover-info-panel")
@app.callback(Output("hover-info-panel", 'children'),
              Input('cytoscape-tsne', 'mouseoverNodeData'))
def displayTapNodeData(data):
    print(data)
    if data:
        spec_id = data['id']
        metdata_information = TSNE_DF.iloc[int(spec_id)].to_dict()
        information_string = (f"spectrum_id: {spec_id}, cluster: {metdata_information['clust']} ")

        return information_string

if __name__ == '__main__':
    app.run_server(debug=True, port="8999")


                                                                               
