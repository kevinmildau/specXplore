# spectrum_plot_panel



@app.callback(
    Output("expand_level", "data"),
    Output("expand_level_input", "placeholder"),
    Input('expand_level_input', 'n_submit'),
    Input("expand_level_input", "value"))

def expand_trigger_handler(n_submit, new_expand_level):
    new_expand_level, new_placeholder=data_transfer.update_expand_level(
        new_expand_level)
    return new_expand_level, new_placeholder

@app.callback( ##################################################################################
    Output('selected-filter-classes', 'data'),
    Input('class-filter-dropdown', 'value')
)
def update_selected_filter_classes(values):
    return values

# GLOBAL OVERVIEW UPDATE TRIGGER
@app.callback(
    Output("tsne-overview-graph", "figure"), 
    Input("tsne-overview-graph", "clickData"), # or "hoverData"
    Input("selected_class_level", "data"),
    Input("selected_class_data", "data"),
    Input('selected-filter-classes', 'data'), 
    Input('specid-focus-dropdown', 'value'),
    State("color_dict", "data"),
    State("tsne-overview-graph", "figure"))

def left_panel_trigger_handler(
    point_selection, 
    selected_class_level, 
    selected_class_data, 
    class_filter_set,
    focus_selection,
    color_dict,
    figure_old):
    """ Modifies global overview plot in left panel """
    #global tsne_fig
    #print(type(figure_old))
    
    triggered_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if (triggered_id == 'tsne-overview-graph' and point_selection):
        print(f'Trigger element: {triggered_id}')
    elif triggered_id == 'selected_class_level':
        print(f'Trigger element: {triggered_id}')
    elif triggered_id == 'selected_class_data':
        print(f'Trigger element: {triggered_id}')
    elif triggered_id == 'selected-filter-classes':
        print(f'Trigger element: {triggered_id}')
    elif triggered_id == 'specid-focus-dropdown' and focus_selection:
        print(f'Trigger element: {triggered_id}')
        tsne_fig=tsne_plotting.plot_tsne_overview(
            point_selection, selected_class_level, selected_class_data, TSNE_DF, 
            class_filter_set, color_dict, focus_selection)
        print(tsne_fig.data[0].marker.color)
    else:
        print("Something else triggered the callback.")
    tsne_fig=tsne_plotting.plot_tsne_overview(
        point_selection, selected_class_level, selected_class_data, TSNE_DF, 
        class_filter_set, color_dict, focus_selection)
    return tsne_fig

# CLASS SELECTION UPDATE ------------------------------------------------------
@app.callback(
    Output("selected_class_level", "data"), 
    Output("selected_class_data", "data"),
    Output("color_dict", "data"), 
    Output('class-filter-dropdown', 'options'), 
    Output('class-filter-dropdown', 'value'),
    Input("class-dropdown", "value"))
def class_update_trigger_handler(selected_class):
    """ Wrapper Function that construct class dcc.store data. """
    selected_class_data, color_dict=data_transfer.update_class(selected_class, 
        CLASS_DICT)
    print("Checkpoint - new selected class data constructed.")
    return selected_class, selected_class_data, color_dict, list(set(selected_class_data)), []





# tsne-overview selection data trigger ----------------------------------------
@app.callback(
    Output('specid-selection-dropdown', 'value'),
    Input('tsne-overview-graph', 'selectedData')
)
def plotly_selected_data_trigger(plotly_selection_data):
    """ Wrapper Function for tsne point selection handling. """
    if ctx.triggered_id == "tsne-overview-graph":
        selected_ids=data_transfer.extract_identifiers_from_plotly_selection(plotly_selection_data)
    else:
        selected_ids = []
    return selected_ids

# Fragmap trigger -------------------------------------------------------------
@app.callback(
    Output('fragmap_panel', 'children'),
    Input('btn_push_fragmap', 'n_clicks'), # trigger only
    State('specid-focus-dropdown', 'value'))
def fragmap_trigger(n_clicks, selection_data):
    """ Wrapper function that calls fragmap generation modules. """
    # Uses: global variable ALL_SPECTRA
    fragmap_panel=fragmap.generate_fragmap_panel(selection_data, ALL_SPECTRA)
    return fragmap_panel












########################################################################################################################
# implement
@app.callback(
    Output('spectrum_plot_panel', 'children'),
    Input('specid-spectrum_plot-dropdown', 'value')) # trigger only
def generate_spectrum_plot(id):
    if id:
        fig = go.Figure(data=[go.Bar(
            x=[1, 2, 3, 5.5, 10],
            y=[10, 8, 6, 4, 2],
            width=[0.8, 0.8, 0.8, 3.5, 4] 
        )])
        fig.update_layout(title="Placeholder Graph spectrum_plot")
        panel = dcc.Graph(id = "spectrum_plot", figure= fig)
        return panel
    else: 
        return html.P("Select spectrum id for plotting.")

########################################################################################################################
# implement
@app.callback(
    Output('mirrorplot_panel', 'children'),
    Input('specid-mirror-1-dropdown', 'value'),
    Input('specid-mirror-2-dropdown', 'value')) # trigger only
def generate_mirror_plot(spectrum_id_1, spectrum_id_2):
    if spectrum_id_1 and spectrum_id_2:
        fig = go.Figure(data=[
            go.Bar(
                x=[1, 2, 3, 5.5, 10],
                y=[10, 8, 6, 4, 2],
                width=[0.8, 0.8, 0.8, 3.5, 4] 
            )])
        fig.update_layout(title="Placeholder Graph mirrorplot")
        panel = dcc.Graph(id = "mirror_plot", figure= fig)
        return panel
    else: 
        return html.P("Select spectrum ids for plotting.")