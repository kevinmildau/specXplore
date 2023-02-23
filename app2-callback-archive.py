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





# NOT USED ANYMORE
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