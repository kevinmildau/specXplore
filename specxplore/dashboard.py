from dash import dcc, html, ctx, dash_table, Dash
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from specxplore import degreemap, egonet, augmap, fragmap, netview, utils, specplot, identifiers, layouts, session_data
from specxplore.constants import COLORS, UNICODE_X
from specxplore.session_data import SpecxploreSessionData
import pickle
import plotly.graph_objects as go
import os
import numpy as np
from typing import Union
from dataclasses import dataclass

@dataclass
class SpecxploreDashboard():
    """ Open Dashboard Instance Class """
    session_data : SpecxploreSessionData
    app : Union[Dash, None] = None

    def __init__ (self, session_data : SpecxploreSessionData):
        assert isinstance(session_data, SpecxploreSessionData), (
            "Error: input is not type SpecxploreSessionData!"
        )
        self.session_data = session_data
        self._initialize_app()
        return None
    
    def run_app(self, **kwargs):
        """ Run the dash server locally. Refer to run_server for input arguments. """
        self.app.run_server(**kwargs)

    def _initialize_app(self):
        # possible themes: VAPOR, UNITED, SKETCHY; see more:  https://bootswatch.com/
        self.app = Dash(external_stylesheets=[dbc.themes.UNITED])
        self.app.layout = html.Div(
            children=[
                layouts.LAYOUT_ROW_TITLE,
                layouts.LAYOUT_ROW_MAIN_PANEL_AND_BUTTONS, 
                layouts.SETTINGS_PANEL,
                layouts.SELECTION_FOCUS_PANEL,
                layouts.LAYOUT_STORE_CONTAINERS,
                layouts.LAYOUT_ROW_MESSAGES_AND_HOVER,
                layouts.LAYOUT_ROW_DETAILS_PANEL,
            ], 
            style={"width" : "100%"},
        )

        @self.app.callback(
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
            styles = self.session_data.initial_style

            elements = node_elements_from_store

            case_highlight_classes =  btn == identifiers.DROPDOWN_CLASSES_TO_BE_HIGHLIGHTED and classes_to_be_highlighted
            case_too_many_classes_to_be_highlighted = (len(classes_to_be_highlighted) > max_colors)

            if case_highlight_classes:
                # define style color update if trigger is class highlight dropdown
                tmp_colors = [{
                    "selector" : f".{str(elem)}", "style" : {"background-color" : COLORS[idx]}} 
                    for idx, elem in enumerate(classes_to_be_highlighted)]
                styles = self.session_data.initial_style + tmp_colors
            if case_too_many_classes_to_be_highlighted:
                warning_messages += (
                    f" \n{UNICODE_X}Number of classes selected = {len(classes_to_be_highlighted)}" 
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
                    self.session_data.tsne_coordinates_table, 
                    spec_id_selection,
                    all_class_level_assignments,
                    threshold, 
                    self.session_data.sources, 
                    self.session_data.targets, 
                    self.session_data.values, 
                    self.session_data.get_spectrum_iloc_list(),
                    max_edges_clustnet, 
                    max_edges_per_node
                )
                if n_omitted_edges != int(0):
                    warning_messages += (
                        f"  \n{UNICODE_X}Current settings (threshold, maximum node degree) lead to edge omission." 
                        f" {n_omitted_edges} edges with lowest edge weight removed from visualization."
                    )
            if case_generate_clustnet_fails_because_no_selection:
                warning_messages += (f"\n{UNICODE_X} No nodes selected, no edges can be rendered.")
            
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
                    self.session_data.sources, 
                    self.session_data.targets, 
                    self.session_data.values, 
                    self.session_data.tsne_coordinates_table, 
                    threshold, 
                    expand_level, 
                    max_edges_egonet
                )
                if n_omitted_edges != int(0):
                    warning_messages += (
                        f"  \n{UNICODE_X}Current settings (threshold, maximum node degree, hop distance) lead to edge omission."
                        f"{n_omitted_edges} edges removed from visualization. These either exceeded maximum node degrees "
                        "in branching tree or were low similarity edges removed to avoid exceeding maximum edge numbers."
                    )
            if case_generate_egonet_fails_because_no_selection:
                warning_messages += (f"  \n{UNICODE_X} No node selected, no edges can be shown.")
            if case_generate_egonet_fails_because_multiselection:
                warning_messages += (
                    f"  \n{UNICODE_X}More than one node selected. Select single spectrum as egonode."
                )

            if btn == identifiers.BUTTON_RUN_DEGREE_MAP:
                styles, legend_plot = degreemap.generate_degree_colored_elements(
                    self.session_data.sources, 
                    self.session_data.targets, 
                    self.session_data.values, 
                    threshold
                )
                if styles and legend_plot:
                    styles = self.session_data.initial_style + styles
                    node_degree_legend = [
                        dcc.Graph(id = "legend", figure = legend_plot, style={"height":"8vh", })
                    ]
                else:
                    styles = self.session_data.initial_style
                    warning_messages += (f"  \n{UNICODE_X} Threshold too stringent. All node degrees are zero.")
            case_change_node_selection_but_keep_style = (
                btn == identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION
                and spec_id_selection
            )
            if case_change_node_selection_but_keep_style:
                styles = self.session_data.initial_style + previous_stylesheet 
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
                styles = self.session_data.initial_style + tmp_colors
            
            return elements, styles, zoom_level, pan_location, node_degree_legend, warning_messages
            

        @self.app.callback(
            Output(identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION, "value"),
            Output(identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION, "options"),
            Input(identifiers.CYTOSCAPE_MAIN_PANEL, "selectedNodeData"),
            Input(identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER, "data"),  
            State(identifiers.DROPDOWN_FOCUS_SPECTRUM_ILOC_SELECTION, "options"),
            prevent_initial_call=True
        )
        def displaySelectedNodeData(
            selection_data, 
            _, 
            old_options):

            btn = ctx.triggered_id
            if btn == identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER:
                new_options = self.session_data.get_spectrum_iloc_list()
                new_focus_id_values = [] # always start wit empty set
                return new_focus_id_values, new_options
            
            if selection_data:
                focus_ids = []
                for elem in selection_data:
                    focus_ids.append(int(elem["id"]))
                return focus_ids, old_options
            else:
                return [], old_options

        @self.app.callback(
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
        @self.app.callback(
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
        @self.app.callback(
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
                panel = fragmap.generate_fragmap_panel(selection_data, self.session_data.spectra_specxplore, top_k_fragmap)
            
            elif (
                btn == identifiers.BUTTON_RUN_METADATA_TABLE 
                and selection_data
                ):
                tmpdf = self.session_data.metadata_table.iloc[selection_data]
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
                    self.session_data.primary_score, 
                    self.session_data.secondary_score , 
                    self.session_data.tertiary_score, 
                    threshold, 
                    colorblind_boolean,
                    self.session_data.score_names
                )
                
            elif btn == identifiers.BUTTON_RUN_SPECPLOT and selection_data and len(selection_data) <=max_number_specplot:
                if len(selection_data) == 1:
                    panel = dcc.Graph(
                        id=identifiers.PANEL_GRAPH_SPECPLOT, 
                        figure=specplot.generate_single_spectrum_plot(
                            self.session_data.spectra_specxplore[selection_data[0]]
                        )
                    )
                if len(selection_data) == 2:
                    panel = dcc.Graph(
                        id=identifiers.PANEL_GRAPH_SPECPLOT, 
                        figure=specplot.generate_mirror_plot(
                            self.session_data.spectra_specxplore[selection_data[0]], 
                            self.session_data.spectra_specxplore[selection_data[1]]
                        )
                    )
                if len(selection_data) > 2:
                    spectra = [self.session_data.spectra_specxplore[i] for i in selection_data]
                    panel = specplot.generate_multiple_spectra_figure_div_list(spectra)
            else:
                panel = []
                warning_message = (
                    f"  \n{UNICODE_X} Insufficient or too many spectra selected for requested details view."
                )
            return panel, warning_message


        @self.app.callback(
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
        @self.app.callback(
            Output(identifiers.PANEL_HOVER_INFO, "children"),
            Input(identifiers.CYTOSCAPE_MAIN_PANEL, "mouseoverNodeData"),
            State(identifiers.DROPDOWN_SELECT_CLASS_LEVEL, "value"),
            prevent_initial_call=True
        )
        def displaymouseoverData(data, selected_class_level):
            """ Callback Function renders class table information and id for hovered over node in text panel."""
            if data:
                spec_id = data["id"]
                node_class_info = self.session_data.classification_table.iloc[int(spec_id)].to_dict()

                selected_class_info = self.session_data.classification_table.iloc[int(spec_id)].to_dict()[selected_class_level]
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


        # expand level control setting
        @self.app.callback(
            Output(identifiers.STORE_HOP_DISTANCE, "data"),
            Output(identifiers.INPUT_HOP_DISTANCE, "placeholder"),
            Input(identifiers.INPUT_HOP_DISTANCE, "n_submit"),
            Input(identifiers.INPUT_HOP_DISTANCE, "value")
        )
        def expand_trigger_handler(_, new_expand_level):
            new_expand_level, new_placeholder=utils.update_expand_level(
                new_expand_level)
            return new_expand_level, new_placeholder

        # expand level control setting
        @self.app.callback(
            Output(identifiers.STORE_MAXIMUM_NODE_DEGREE, "data"),
            Output(identifiers.INPUT_MAXIMUM_NODE_DEGREE, "placeholder"),
            Input(identifiers.INPUT_MAXIMUM_NODE_DEGREE, "n_submit"),
            Input(identifiers.INPUT_MAXIMUM_NODE_DEGREE, "value")
        )
        def max_degree_trigger_handler(_, new_max_degree):
            new_max_degree, new_placeholder=utils.update_max_degree(new_max_degree)
            return new_max_degree, new_placeholder

        # CLASS SELECTION UPDATE ------------------------------------------------------
        @self.app.callback(
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
            class_levels = list(self.session_data.class_dict.keys())
            btn = ctx.triggered_id
            # Update selected_class if the trigger for class updating is a new session data.
            if btn == identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER:
                selected_class = list(self.session_data.class_dict.keys())[0]
            class_assignments = self.session_data.class_dict[selected_class]
            unique_assignments = list(np.unique(class_assignments))
            node_elements = utils.initialize_cytoscape_graph_elements(
                self.session_data.tsne_coordinates_table, 
                class_assignments, 
                self.session_data.highlight_table["highlight_bool"].to_list()
            )
            return selected_class, class_levels, class_assignments, unique_assignments, [], node_elements, selected_class


        @self.app.callback(
            Output(identifiers.FIGURE_EDGE_WEIGHT_HISTOGRAM, "figure"),
            Input(identifiers.STORE_EDGE_THRESHOLD, "data"),
            Input(identifiers.STORE_EMPTY_SESSION_DATA_TRIGGER, "data")
        )
        def update_histogram(
                threshold : float, _ : None
                ) -> go.Figure:
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                y=self.session_data.values, opacity = 0.6, 
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

        @self.app.callback(
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
                The global variable self.session_data is modified by this function.
            
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
            # if case_update_scaling or case_upload_new_session_data:
            #     self.session_data
            if case_update_scaling:
                self.session_data.scale_coordinate_system(scaler)
                print("Coordinate system scaled.")
            elif case_upload_new_session_data:
                # check for valid and existing file
                # load file
                with open(filename, "rb") as handle:
                    specxplore_object = pickle.load(handle) 
                # assess compatibility of output
                if isinstance(specxplore_object, session_data.SpecxploreSessionData):
                    self.session_data = specxplore_object
                    self.session_data.scale_coordinate_system(scaler)
                    print("Session data updated.")
                else: 
                    print("Input file wrong format. No data update.")
            else:
                ...
            return {"None": None}

        @self.app.callback(
            Output(identifiers.STORE_COLORBLIND_BOOLEAN, "data"),
            Input(identifiers.SWITCH_AUGMAP_COLORBLIND_CHANGE, "on")
        )
        def update_output(on_off_state):
            return on_off_state
        return None