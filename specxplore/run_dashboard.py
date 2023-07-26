from dash import dcc, html, ctx, dash_table, Dash
from dash.dependencies import Input, Output, State
import dash_daq as daq
import dash_bootstrap_components as dbc
import dash_cytoscape as cyto
import specxplore
from specxplore import degreemap, egonet, augmap, fragmap, netview, utils, specplot, identifiers
import specxplore.importing
from specxplore.constants import COLORS

import pickle
import plotly.graph_objects as go
import os
import numpy as np
from typing import Union

# Load Initial Data State (to be migrated to within package data structure)
specxplore_input_file = "/Users/kevinmildau/Dropbox/univie/Project embedding a molecular network/development/specxplore-illustrative-examples/data/data_wheat_output/specxplore_wheat.pickle"
with open(specxplore_input_file, "rb") as handle:
    GLOBAL_DATA = pickle.load(handle) 
    GLOBAL_DATA.scale_coordinate_system(400)



# All settings panel
settings_panel = dbc.Offcanvas(
    children=[
        html.B("Set Edge Threshold:"),
        html.P("A value between 0 and 1 (excluded)"),
        dcc.Input(
            id = identifiers.INPUT_EDGE_THRESHOLD, 
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
                    id = identifiers.FIGURE_EDGE_WEIGHT_HISTOGRAM, 
                    figure = [],  
                    style={"width":"100%","height":"30vh"}
                )
            ]
        ),
        html.Br(),
        html.B("Set maximum node degree:"),
        html.P("A value between 1 and 9999. The lower, the fewer edges are allowed for each node."),
        dcc.Input(
            id=identifiers.INPUT_MAXIMUM_NODE_DEGREE, 
            type="number", 
            debounce=True, 
            placeholder="1 <= Value <= 9999, def. 9999", 
            style={"width" : "100%"}
        ),
        html.Br(),
        html.B("Higlight Class(es) by color:"),
        dcc.Dropdown(
            id=identifiers.DROPDOWN_CLASSES_TO_BE_HIGHLIGHTED , 
            multi=True, 
            clearable=False, 
            options=[],
            value=[]
        ),
        html.Br(),
        html.B("Set Hop Distance for EgoNet:"),
        html.P("A value between 1 and 5. Controls connection branching in EgoNet."),
        dcc.Input(
            id=identifiers.INPUT_HOP_DISTANCE, 
            type="number", 
            debounce=True, 
            placeholder="Value between 1 < exp.lvl. < 5, def. 1", 
            style={"width" : "100%"}
        ),
        html.Br(),
        html.B("Select Class Level:"),
        dcc.Dropdown( 
            id=identifiers.DROPDOWN_SELECT_CLASS_LEVEL, 
            multi=False, 
            clearable=False, 
            options= GLOBAL_DATA.available_classes, 
            value=GLOBAL_DATA.available_classes[0]
        ),
        html.Br(),
        html.B("Toggle colorblind friendly for Augmap:"),
        daq.BooleanSwitch(
            id=identifiers.SWITCH_AUGMAP_COLORBLIND_CHANGE, 
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
            id=identifiers.INPUT_TOP_K_FRAGMENT_LIMIT_FOR_FRAGMAP, type="number", 
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
            id=identifiers.INPUT_COORDINATE_SCALING, 
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
                    id=identifiers.INPUT_UPLOAD_NEW_SESSION_DATA, 
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
    id=identifiers.PANEL_OFFCANVAS_SETTINGS,
    title="specXplore settings",
    is_open=False)


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
            id=identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION, 
            multi=True, 
            style={"width": "90%", "font-size": "100%"}, 
            options=GLOBAL_DATA.get_spectrum_iloc_list())
    ],
    id=identifiers.PANEL_OFFCANVAS_FOCUS_SPECTRUM_ILOC_SELECTION,
    placement="end",
    title="Selection Panel",
    is_open=False)


# Button array on left side of plot

BUTTON_HEIGHT = "9vh"
BUTTON_STYLE = {
    "width": "98%", 
    "height": BUTTON_HEIGHT, 
    "fontSize": "11px", 
    "textAlign": "center", 
    "border":"1px black solid",
    "backgroundColor" : "#8B008B"
}

control_button_group = [
    dbc.ButtonGroup(
        children=[
            dbc.Button("⚙ Settings", id=identifiers.BUTTON_OPEN_SETTINGS, style = BUTTON_STYLE),
            dbc.Button("View Selection", id=identifiers.BUTTON_OPEN_FOCUS, style = BUTTON_STYLE),
            dbc.Button("Show Node Degree ", id=identifiers.BUTTON_RUN_DEGREE_MAP, n_clicks=0, style = BUTTON_STYLE),
            dbc.Button("EgoNet ", id=identifiers.BUTTON_RUN_EGONET, n_clicks=0, style = BUTTON_STYLE),
            dbc.Button("Show Edges for Selection", id=identifiers.BUTTON_RUN_NETVIEW, n_clicks=0, style = BUTTON_STYLE),  
            dbc.Button("Augmap", id=identifiers.BUTTON_RUN_AUGMAP, n_clicks=0, style = BUTTON_STYLE),
            dbc.Button("Spectrum Plot(s)", id=identifiers.BUTTON_RUN_SPECPLOT, n_clicks=0, style = BUTTON_STYLE),
            dbc.Button("Fragmap", id=identifiers.BUTTON_RUN_FRAGMAP, n_clicks=0, style = BUTTON_STYLE),
            dbc.Button("Metadata Table", id=identifiers.BUTTON_RUN_METADATA_TABLE, n_clicks=0, style = BUTTON_STYLE)
        ],
        vertical=True, 
        style = {"width" : "100%"}
    )
]


# possible themes: VAPOR, UNITED, SKETCHY; see more:  https://bootswatch.com/
app=Dash(external_stylesheets=[dbc.themes.UNITED])



title_row = dbc.Row(
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
)

main_panel_and_buttons_row = dbc.Row(
    children = [
        dbc.Col(control_button_group, width = 2),
        dbc.Col(
            children = [
                cyto.Cytoscape( 
                    id=identifiers.CYTOSCAPE_MAIN_PANEL, 
                    elements=GLOBAL_DATA.initial_node_elements, 
                    stylesheet = GLOBAL_DATA.initial_style,
                    layout={
                        "name":"preset", 
                        "animate":False, 
                        "fit": False}, 
                    boxSelectionEnabled=True,
                    style={
                        "width":"100%", 
                        "height":"80vh", 
                        "border":"1px black solid", 
                        "bg":"#feeff4", 
                        "minZoom":0.1, 
                        "maxZoom":2})
            ], 
            width=10
        ) 
    ], 
    style={"margin": "0px"}, 
    className="g-0"
)

messages_and_hover_info_row = dbc.Row(
    children = [
        dbc.Col(
            children = [
                html.Tbody("Hover Info: placeholder for cyto hover information")
            ], 
            id = identifiers.PANEL_HOVER_INFO,
            style={"font-size" : "8pt", "border":"1px black solid"}),
        dbc.Col(
            children = [
                dcc.Markdown(
                    id=identifiers.PANEL_WARNING_MESSAGES_1, 
                    style={
                        "font-size" : "8pt", 
                        "border":"1px black solid"
                    }
                ),
                dcc.Markdown(
                    id=identifiers.PANEL_WARNING_MESSAGES_2, 
                    style={
                        "font-size" : "8pt", 
                        "border":"1px black solid"
                    }
                )
            ], 
            width=4), 
        dbc.Col(
            children = [
                html.Div(
                    id=identifiers.PANEL_NODE_DEGREE_LEGEND, 
                    style={
                        "width":"100%", 
                        "border":"1px black solid"
                    }
                )
            ], 
            width=8)
        ], 
    className="g-0"
)

details_panel_row = dbc.Row(
    children = [
        dbc.Col(
            children = [
                html.Div(
                    id=identifiers.PANEL_DETAIL_VISUALIZATIONS, 
                    style={
                        "width":"100%", 
                        "height":"90vh", 
                        "border":"1px black solid"
                    }
                )
            ], 
            width=12)
    ], 
    className="g-0"
)

app.layout=html.Div(
    children=[
        title_row,
        main_panel_and_buttons_row, 
        settings_panel,
        selection_focus_panel,
        dcc.Store(id=identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER, data = None),
        dcc.Store(id=identifiers.STORE_EDGE_THRESHOLD, data=0.9),
        dcc.Store(id=identifiers.STORE_HOP_DISTANCE, data=int(1)),
        dcc.Store(id=identifiers.STORE_MAXIMUM_NODE_DEGREE, data = int(9999)),
        dcc.Store(id=identifiers.STORE_SELECTED_CLASS_LEVEL, data = GLOBAL_DATA.available_classes[0]),
        dcc.Store(id=identifiers.STORE_CLASSES_TO_BE_HIGHLIGHTED, data=[GLOBAL_DATA.available_classes[0]]),
        dcc.Store(id=identifiers.STORE_NODE_ELEMENTS, data = GLOBAL_DATA.initial_node_elements),
        dcc.Store(id=identifiers.STORE_COLORBLIND_BOOLEAN, data = False),
        #html.Br(),
        messages_and_hover_info_row,
        details_panel_row,
    ], 
    style={"width" : "100%"},
)

# cytoscape update callbacks

@app.callback(
    Output(identifiers.CYTOSCAPE_MAIN_PANEL, "elements"),
    Output(identifiers.CYTOSCAPE_MAIN_PANEL, "stylesheet"),
    Output(identifiers.CYTOSCAPE_MAIN_PANEL, "zoom"),
    Output(identifiers.CYTOSCAPE_MAIN_PANEL, "pan"),
    Output(identifiers.PANEL_NODE_DEGREE_LEGEND, "children"),
    Output(identifiers.PANEL_WARNING_MESSAGES_1, "children"),
    Input(identifiers.BUTTON_RUN_EGONET, "n_clicks"),
    Input(identifiers.BUTTON_RUN_NETVIEW, "n_clicks"),
    Input(identifiers.BUTTON_RUN_DEGREE_MAP, "n_clicks"),
    Input(identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION, "value"), 
    Input(identifiers.DROPDOWN_CLASSES_TO_BE_HIGHLIGHTED, "value"),
    Input(identifiers.STORE_CLASSES_TO_BE_HIGHLIGHTED, "data"),
    Input(identifiers.STORE_NODE_ELEMENTS, "data"),
    Input(identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER, "data"),  
    State(identifiers.STORE_MAXIMUM_NODE_DEGREE, "data"),
    State(identifiers.STORE_EDGE_THRESHOLD, "data"),
    State(identifiers.STORE_HOP_DISTANCE, "data"),
    State(identifiers.CYTOSCAPE_MAIN_PANEL, "zoom"),
    State(identifiers.CYTOSCAPE_MAIN_PANEL, "pan"),
    State(identifiers.CYTOSCAPE_MAIN_PANEL, "elements"),   # <-- for persistency
    State(identifiers.CYTOSCAPE_MAIN_PANEL, "stylesheet"), # <-- for persistency
    prevent_initial_call=True
)
def cytoscape_trigger(
        _n_clicks1, 
        _n_clicks2,
        _n_clicks3,
        spec_id_selection, 
        classes_to_be_highlighted,
        all_class_level_assignments, 
        node_elements_from_store,
        _none_data_trigger, 
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
    max_edges_egonet = 2500 # currently not used: make an input state, make it actually used

    node_degree_legend = [] # initialize to empty list. Only get filled if degree node selected
    warning_messages = "" # initialize to empty string. Only gets expanded if warnings necessary.
    styles = GLOBAL_DATA.initial_style

    elements = node_elements_from_store

    case_highlight_classes =  btn == identifiers.DROPDOWN_CLASSES_TO_BE_HIGHLIGHTED and classes_to_be_highlighted
    case_too_many_classes_to_be_highlighted = (len(classes_to_be_highlighted) > max_colors)

    if case_highlight_classes:
        # define style color update if trigger is class highlight dropdown
        tmp_colors = [{
            "selector" : f".{str(elem)}", "style" : {"background-color" : COLORS[idx]}} 
            for idx, elem in enumerate(classes_to_be_highlighted)]
        styles = GLOBAL_DATA.initial_style + tmp_colors
    if case_too_many_classes_to_be_highlighted:
        warning_messages += (
            f" \n❌Number of classes selected = {len(classes_to_be_highlighted)}" 
            f"exceeds available colors = {max_colors}."
            " Please select fewer classes for highlighting.")
        classes_to_be_highlighted = classes_to_be_highlighted[0:8]
    
    case_generate_clustnet = (
        btn == identifiers.BUTTON_RUN_NETVIEW 
        and spec_id_selection
    )
    case_generate_clustnet_fails_because_no_selection = (
        btn == identifiers.BUTTON_RUN_NETVIEW 
        and not spec_id_selection
    )
    
    if case_generate_clustnet:
        elements, styles, n_omitted_edges = netview.generate_cluster_node_link_diagram_cythonized(
            GLOBAL_DATA.tsne_coordinates_table, 
            spec_id_selection,
            GLOBAL_DATA.scores_ms2deepscore, 
            all_class_level_assignments,
            threshold, 
            GLOBAL_DATA.sources, 
            GLOBAL_DATA.targets, 
            GLOBAL_DATA.values, 
            GLOBAL_DATA.get_spectrum_iloc_list(),
            max_edges_clustnet, max_edges_per_node
        )
        if n_omitted_edges != int(0):
            warning_messages += (
                f"  \n❌Current settings (threshold, maximum node degree) lead to edge omission." 
                f" {n_omitted_edges} edges with lowest edge weight removed from visualization."
            )
    if case_generate_clustnet_fails_because_no_selection:
        warning_messages += (f"\n❌ No nodes selected, no edges can be rendered.")
    
    case_generate_egonet = (
        btn == identifiers.BUTTON_RUN_EGONET 
        and spec_id_selection 
        and len(spec_id_selection) == 1
    )
    case_generate_egonet_fails_because_no_selection = (
        btn == identifiers.BUTTON_RUN_EGONET 
        and not spec_id_selection
    )
    case_generate_egonet_fails_because_multiselection = (
        btn == identifiers.BUTTON_RUN_EGONET 
        and spec_id_selection 
        and len(spec_id_selection)>1
    )
    if case_generate_egonet:
        elements, styles, n_omitted_edges = egonet.generate_egonet_cythonized(
            spec_id_selection, 
            GLOBAL_DATA.sources, 
            GLOBAL_DATA.targets, 
            GLOBAL_DATA.values, 
            GLOBAL_DATA.tsne_coordinates_table, 
            threshold, 
            expand_level, 
            max_edges_egonet
        )
        if n_omitted_edges != int(0):
            warning_messages += (
                f"  \n❌Current settings (threshold, maximum node degree, hop distance) lead to edge omission."
                f"{n_omitted_edges} edges removed from visualization. These either exceeded maximum node degrees "
                "in branching tree or were low similarity edges removed to avoid exceeding maximum edge numbers."
            )
    if case_generate_egonet_fails_because_no_selection:
        warning_messages += (f"  \n❌ No node selected, no edges can be shown.")
    if case_generate_egonet_fails_because_multiselection:
        warning_messages += (
            f"  \n❌More than one node selected. Select single spectrum as egonode."
        )

    if btn == identifiers.BUTTON_RUN_DEGREE_MAP:
        styles, legend_plot = degreemap.generate_degree_colored_elements(
            GLOBAL_DATA.sources, 
            GLOBAL_DATA.targets, 
            GLOBAL_DATA.values, 
            threshold
        )
        if styles and legend_plot:
            styles = GLOBAL_DATA.initial_style + styles
            node_degree_legend = [
                dcc.Graph(id = "legend", figure = legend_plot, style={"height":"8vh", })
            ]
        else:
            styles = GLOBAL_DATA.initial_style
            warning_messages += (f"  \n❌ Threshold too stringent. All node degrees are zero.")
    case_change_node_selection_but_keep_style = (
        btn == identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION
        and spec_id_selection
    )
    if case_change_node_selection_but_keep_style:
        styles = GLOBAL_DATA.initial_style + previous_stylesheet 
        elements = previous_elements
    
    case_highlight_selected_classes_unless_other_view_requested = (
        btn == identifiers.DROPDOWN_CLASSES_TO_BE_HIGHLIGHTED 
        and not case_generate_clustnet
        and not case_generate_egonet
        or (not btn in (identifiers.BUTTON_RUN_DEGREE_MAP))
        and classes_to_be_highlighted 
        and not spec_id_selection
    )
    if case_highlight_selected_classes_unless_other_view_requested:
        tmp_colors = [
            {
                "selector" : f".{str(elem)}", 
                "style" : {"background-color" : COLORS[idx]}
            } 
            for idx, elem 
            in enumerate(classes_to_be_highlighted)
        ]
        styles = GLOBAL_DATA.initial_style + tmp_colors
    
    return elements, styles, zoom_level, pan_location, node_degree_legend, warning_messages
    

# Set focus ids and open focus menu
@app.callback(
    Output(identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION, "value"),
    Output(identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION, "options"),
    Input(identifiers.CYTOSCAPE_MAIN_PANEL, "selectedNodeData"),
    Input(identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER, "data"),  
    State(identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION, "options"),
    #prevent_initial_call=True
)
def displaySelectedNodeData(
    selection_data, 
    _, 
    old_options):

    btn = ctx.triggered_id
    if btn == identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER:
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
    Output(identifiers.PANEL_OFFCANVAS_FOCUS_SPECTRUM_ILOC_SELECTION, "is_open"),
    Input(identifiers.BUTTON_OPEN_FOCUS, "n_clicks"),
    State(identifiers.PANEL_OFFCANVAS_FOCUS_SPECTRUM_ILOC_SELECTION, "is_open"),
    prevent_initial_call=True
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


# open settings panel
@app.callback(
    Output(identifiers.PANEL_OFFCANVAS_SETTINGS, "is_open"),
    Input(identifiers.BUTTON_OPEN_SETTINGS, "n_clicks"),
    State(identifiers.PANEL_OFFCANVAS_SETTINGS, "is_open"),
    prevent_initial_call=True
)
def toggle_offcanvas(n1, is_open):
    if n1:
        return not is_open
    return is_open


# details panel triggers
@app.callback(
    Output(identifiers.PANEL_DETAIL_VISUALIZATIONS, "children"),
    Output(identifiers.PANEL_WARNING_MESSAGES_2, "children"),
    Input(identifiers.BUTTON_RUN_FRAGMAP, "n_clicks"), 
    Input(identifiers.BUTTON_RUN_METADATA_TABLE, "n_clicks"),
    Input(identifiers.BUTTON_RUN_AUGMAP, "n_clicks"), 
    Input(identifiers.BUTTON_RUN_SPECPLOT, "n_clicks"),
    State(identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION, "value"),
    State(identifiers.STORE_EDGE_THRESHOLD, "data"),
    State(identifiers.STORE_COLORBLIND_BOOLEAN, "data"),
    State(identifiers.INPUT_TOP_K_FRAGMENT_LIMIT_FOR_FRAGMAP, "value"),
    prevent_initial_call=True
)
def details_trigger(
        _btn_fragmap_n_clicks, 
        _btn_meta_n_clicks, 
        _btn_augmap_n_clicks, 
        _btn_spectrum_n_clicks, 
        selection_data, 
        threshold,
        colorblind_boolean, top_k_fragmap):
    """ Wrapper function that calls fragmap generation modules. """
    warning_message = ""
    btn = ctx.triggered_id
    max_number_augmap = 500
    max_number_fragmap = 40
    max_number_specplot = 25

    if (
        btn == identifiers.BUTTON_RUN_FRAGMAP 
        and selection_data 
        and len(selection_data) >= 2 
        and len(selection_data) <= max_number_fragmap
        ):
        panel = fragmap.generate_fragmap_panel(selection_data, GLOBAL_DATA.spectra, top_k_fragmap)
    
    elif (
        btn == identifiers.BUTTON_RUN_METADATA_TABLE 
        and selection_data
        ):
        tmpdf = GLOBAL_DATA.metadata_table.iloc[selection_data]
        panel = dash_table.DataTable(
            id=identifiers.PANEL_METADATA_TABLE,
            columns=[
                {"name": i, "id": i} 
                for i in tmpdf.columns
            ],
            data=tmpdf.to_dict("records"),
            style_cell=dict(textAlign="left"),
            style_header=dict(
                backgroundColor="#8B008B", 
                color="white", 
                border = "1px solid black" 
            ),
            #style_data=dict(backgroundColor="white"),
            sort_action="native",
            page_size=10,
            style_table={"overflowX": "auto"},
        )
    
    elif (
        btn == identifiers.BUTTON_RUN_AUGMAP 
        and selection_data 
        and len(selection_data) >= 2 
        and len(selection_data) <= max_number_augmap
        ):
        panel = augmap.generate_augmap_panel(
            selection_data, 
            GLOBAL_DATA.scores_ms2deepscore, 
            GLOBAL_DATA.scores_modified_cosine , 
            GLOBAL_DATA.scores_spec2vec, 
            threshold, colorblind_boolean
        )
        
    elif btn == identifiers.BUTTON_RUN_SPECPLOT and selection_data and len(selection_data) <=max_number_specplot:
        if len(selection_data) == 1:
            panel = dcc.Graph(
                id=identifiers.PANEL_GRAPH_SPECPLOT, 
                figure=specplot.generate_single_spectrum_plot(
                    GLOBAL_DATA.spectra[selection_data[0]]
                )
            )
        if len(selection_data) == 2:
            panel = dcc.Graph(
                id=identifiers.PANEL_GRAPH_SPECPLOT, 
                figure=specplot.generate_mirror_plot(
                    GLOBAL_DATA.spectra[selection_data[0]], 
                    GLOBAL_DATA.spectra[selection_data[1]]
                )
            )
        if len(selection_data) > 2:
            spectra = [GLOBAL_DATA.spectra[i] for i in selection_data]
            panel = specplot.generate_multiple_spectra_figure_div_list(spectra)
    else:
        panel = []
        warning_message = (
            "  \n❌ Insufficient or too many spectra selected for requested details view."
        )
    return panel, warning_message


@app.callback(
    Output(identifiers.STORE_EDGE_THRESHOLD, "data"),
    Output(identifiers.INPUT_EDGE_THRESHOLD, "placeholder"),
    Input(identifiers.INPUT_EDGE_THRESHOLD, "n_submit"),
    Input(identifiers.INPUT_EDGE_THRESHOLD, "value"),
    prevent_initial_call=True
)
def update_threshold_trigger_handler(_n_submit, new_threshold):
    new_threshold, new_placeholder=utils.update_threshold(new_threshold)
    return new_threshold, new_placeholder


# mouseover node information display
# dbc.Col([html.P("Hover Info: placeholder for cyto hover information")], id = identifiers.PANEL_HOVER_INFO)
@app.callback(
    Output(identifiers.PANEL_HOVER_INFO, "children"),
    Input(identifiers.CYTOSCAPE_MAIN_PANEL, "mouseoverNodeData"),
    State(identifiers.DROPDOWN_SELECT_CLASS_LEVEL, "value"),
    prevent_initial_call=True
)
def displaymouseoverData(data, selected_class_level):
    """ Callback Function renders class table information and id for hovered over node in text panel."""
    if data:
        spec_id = data["id"]
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

#
# expand level control setting
@app.callback(
    Output(identifiers.STORE_HOP_DISTANCE, "data"),
    Output(identifiers.INPUT_HOP_DISTANCE, "placeholder"),
    Input(identifiers.INPUT_HOP_DISTANCE, "n_submit"),
    Input(identifiers.INPUT_HOP_DISTANCE, "value"))

def expand_trigger_handler(_, new_expand_level):
    new_expand_level, new_placeholder=utils.update_expand_level(
        new_expand_level)
    return new_expand_level, new_placeholder

# expand level control setting
@app.callback(
    Output(identifiers.STORE_MAXIMUM_NODE_DEGREE, "data"),
    Output(identifiers.INPUT_MAXIMUM_NODE_DEGREE, "placeholder"),
    Input(identifiers.INPUT_MAXIMUM_NODE_DEGREE, "n_submit"),
    Input(identifiers.INPUT_MAXIMUM_NODE_DEGREE, "value"))

def max_degree_trigger_handler(_, new_max_degree):
    new_max_degree, new_placeholder=utils.update_max_degree(new_max_degree)
    return new_max_degree, new_placeholder



# CLASS SELECTION UPDATE ------------------------------------------------------
@app.callback(
    Output(identifiers.STORE_SELECTED_CLASS_LEVEL, "data"), 
    Output(identifiers.DROPDOWN_SELECT_CLASS_LEVEL, "options"), 
    Output(identifiers.STORE_CLASSES_TO_BE_HIGHLIGHTED, "data"),
    Output(identifiers.DROPDOWN_CLASSES_TO_BE_HIGHLIGHTED, "options"), 
    Output(identifiers.DROPDOWN_CLASSES_TO_BE_HIGHLIGHTED, "value"),
    Output(identifiers.STORE_NODE_ELEMENTS, "data"),
    Output(identifiers.DROPDOWN_SELECT_CLASS_LEVEL, "value"),
    Input(identifiers.DROPDOWN_SELECT_CLASS_LEVEL, "value"),
    Input(identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER, "data") 
)
def class_update_trigger_handler(
        selected_class : str, 
        _ : None):
    """ Wrapper Function that construct class dcc.store data. """
    class_levels = list(GLOBAL_DATA.class_dict.keys())
    btn = ctx.triggered_id
    # Update selected_class if the trigger for class updating is a new session data.
    if btn == identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER:
        selected_class = list(GLOBAL_DATA.class_dict.keys())[0]
    class_assignments = GLOBAL_DATA.class_dict[selected_class]
    unique_assignments = list(np.unique(class_assignments))
    node_elements = utils.initialize_cytoscape_graph_elements(
        GLOBAL_DATA.tsne_coordinates_table, 
        class_assignments, 
        GLOBAL_DATA.highlight_table["highlight_bool"].to_list()
    )
    return selected_class, class_levels, class_assignments, unique_assignments, [], node_elements, selected_class


@app.callback(
    Output(identifiers.FIGURE_EDGE_WEIGHT_HISTOGRAM, "figure"),
    Input(identifiers.STORE_EDGE_THRESHOLD, "data"),
    Input(identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER, "data")
)
def update_histogram(
        threshold : float, _ : None
        ) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        y=GLOBAL_DATA.values, opacity = 0.6, 
        ybins=dict(start=0, end=1,size=0.05), marker_color = "grey")) #, histnorm="percent"
    fig.add_hline(y=threshold, line_dash = "dash", line_color = "magenta", line_width = 5)
    fig.update_traces(marker_line_width=1,marker_line_color="black")
    fig.update_layout(
        barmode="group", bargap=0, bargroupgap=0.0, yaxis = {"range" :[-0.01,1.01]})
    fig.update_layout(
        template="simple_white", 
        title= "Edge Weight Distibution with Threshold",
        xaxis_title="Count", yaxis_title="Edge Weight Bins",
        margin = {"autoexpand":True, "b" : 10, "l":10, "r":10, "t":40})
    return fig


@app.callback(
    Output(identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER, "data"),
    Input(identifiers.INPUT_UPLOAD_NEW_SESSION_DATA, "value"),
    Input(identifiers.INPUT_COORDINATE_SCALING, "value")
)
def update_session_data(filename : str, scaler : Union[int, float]) -> dict:    
    """
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
    2) If the upload triggers, update the specXplore by importing data and using input-scaler-id value
    """
    # 
    trigger_component = ctx.triggered_id

    case_upload_new_session_data = (
        trigger_component == identifiers.INPUT_UPLOAD_NEW_SESSION_DATA 
        and filename is not None 
        and os.path.isfile(filename))
    case_update_scaling = (
        trigger_component == identifiers.INPUT_COORDINATE_SCALING 
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
        with open(filename, "rb") as handle:
            specxplore_object = pickle.load(handle) 
        # assess compatibility of output
        if isinstance(specxplore_object, specxplore.importing.specxplore_session):
            GLOBAL_DATA = specxplore_object
            GLOBAL_DATA.scale_coordinate_system(scaler)
            print("Session data updated.")
        else: 
            print("Input file wrong format. No data update.")
    else:
        ...
    return {"None": None}


@app.callback(
    Output(identifiers.STORE_COLORBLIND_BOOLEAN, "data"),
    Input(identifiers.SWITCH_AUGMAP_COLORBLIND_CHANGE, "on")
)
def update_output(on_off_state):
    return on_off_state

if __name__ == "__main__":    
    app.run(debug=True, port="8999")