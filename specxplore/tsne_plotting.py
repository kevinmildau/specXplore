# Developer Notes
# for more complex plot specifications, use go.Scattergl() rather than px

# Note that the current implementation favors the focus selection color over the
# cluster selection; a point already in focus selection cannot be clicked for
# clust recoloring, nor will it be recolored if part of a cluster with that color.

import plotly.express as px
import plotly.graph_objects as go
import copy
def plot_tsne_overview(
    point_selection, selected_class_level, selected_class_data, tsne_df, 
    class_filter_set, color_dict, focus_selection):
    """Constructs the t-SNE overview graph plotly figure object.
    
    Args / Parameters
    ------
    point_selection:
        None, a single selected spec_id, or a multi-selection of spec_ids. The
        selection is used to highlight the cluster to which a selected spec_id
        belongs. If a multi-selection is provided, the first indexed spec_id
        is used for cluster highlighting.
    selected_class_level: 
        Name string of the selected class. 
    selected_class_data:
        The list of class assignments for the selected class level.
    tsne_df:
        Dataframe with t-SNE coordinates for each spec_id.
    color_dict:
        A color dictionary where each unique cluster is a string key with the
        corresponding value being a hex code color: {clust_key : color_string}.
    class_filter_set:
        Set of classes with which the dataset is to be filtered prior to
        visualization. Assists in showing meaningful subsets of the data only.
    Returns
    ------
    fig:
        t-SNE overview graph as ploty figure object.
    """
    # Modify color dict to highlight class of selected spectrum; if multiple
    # selected the first indexed spectrum is highlighted.
    if point_selection:  
        selected_point=point_selection["points"][0]["customdata"][0]
        color_dict[selected_class_data[selected_point]]="#FF10F0"
    
        
    # Extend df to contain selected class data (always given)
    tmpdf = copy.deepcopy(tsne_df)
    tmpdf["clust"]=selected_class_data
    tmpdf["color"]=selected_class_data
    
    if focus_selection:
        color_iloc =  tmpdf.columns.get_loc('color')
        tmpdf.iloc[focus_selection, color_iloc] = "focus_selection"
        color_dict["focus_selection"] = "#30D5C8"
    
    if class_filter_set:
        class_filter_mask = tmpdf['clust'].isin(class_filter_set)
        tmpdf = tmpdf.loc[class_filter_mask]

    standards_filter_mask = tmpdf['is_standard'] == True
    tmpdf_standards = tmpdf.loc[standards_filter_mask]
    
    spectra_filter_mask = tmpdf['is_standard'] == False
    tmpdf_spectra = tmpdf.loc[spectra_filter_mask]

    fig1=px.scatter(
        tmpdf_spectra, 
        x="x", y="y", color="color", 
        custom_data=["id", "clust"], 
        color_discrete_map=color_dict, 
        render_mode='webgl')
    fig1.update_traces(
        marker={
            'size': 8,
            'line':dict(width=0.8,color='DarkSlateGrey'),
            'opacity': 0.7 },
        hovertemplate='Spec ID: %{customdata[0]}<br>Cluster/Class: %{customdata[1]}<extra></extra>'
        )
    fig2=px.scatter(
        tmpdf_standards, 
        x="x", y="y", color="color", 
        custom_data=["id", "clust"], 
        color_discrete_map=color_dict, 
        render_mode='webgl')
    fig2.update_traces(
        marker={
            'symbol':220,
            'size': 8,
            'line':dict(width=1,color='black'),
            'opacity': 0.8 },
        hovertemplate='Spec ID: %{customdata[0]}<br>Cluster/Class: %{customdata[1]}<extra></extra>'
        )
    fig = go.Figure()
    fig.add_traces(fig1.data)
    fig.add_traces(fig2.data)
    
    
    fig.update_layout(
        yaxis_visible=False, yaxis_showticklabels=False, 
        xaxis_visible=False, xaxis_showticklabels=False, 
        title_text='T-SNE Spectral Data Overview', 
        title_x=0.01, title_y=0.01,
        legend_title_text=selected_class_level,
        paper_bgcolor='white', # black or white interfere with grayscale
        plot_bgcolor='white', # black or white interfere with grayscale
        uirevision="Never", # prevent zoom level changes upon plot update
        legend=dict(title_font_family='Ubuntu',font=dict(size=6)),
        showlegend=False,
        clickmode='event', 
        hovermode="closest",
        margin={"autoexpand":True, "b" : 0, "l":0, "r":50, "t":0},
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Ubuntu"))

    return fig