# Developer Notes
# for more complex plot specifications, use go.Scattergl() rather than px

# Settings
# {tsne_overview : 
# {highlight_color : "#FF10F0",  paper_bgcolor :'#feeff4', plot_bgcolor='#feeff4'}}

import plotly.express as px
def plot_tsne_overview(
    point_selection, selected_class_level, selected_class_data, tsne_df, 
    class_filter_set, color_dict):
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
    tsne_df_tmp=tsne_df
    tsne_df_tmp["clust"]=selected_class_data
    
    if class_filter_set:
        print("Problem", class_filter_set)
        tsne_df_tmp = tsne_df_tmp[tsne_df_tmp["clust"].isin(class_filter_set)]

    fig=px.scatter(
        tsne_df_tmp, 
        x="x", y="y", color="clust", 
        custom_data=["id"], 
        color_discrete_map=color_dict, 
        render_mode='webgl')
    fig.update_layout(
        clickmode='event+select', 
        margin={"autoexpand":True, "b" : 0, "l":0, "r":50, "t":0})
    fig.update_layout(
        yaxis_visible=False, yaxis_showticklabels=False, 
        xaxis_visible=False, xaxis_showticklabels=False, 
        title_text='T-SNE Spectral Data Overview', 
        title_x=0.01, title_y=0.01,
        legend_title_text=selected_class_level,
        paper_bgcolor='#feeff4', # black or white interfere with grayscale
        plot_bgcolor='#feeff4', # black or white interfere with grayscale
        uirevision=False, # prevent zoom level changes upon plot update
        legend=dict(title_font_family='Ubuntu',font=dict(size=6)),
        showlegend=False)
    fig.update_traces(
        marker={
            'size': 8,
            'line':dict(width=0.8,color='DarkSlateGrey'),
            'opacity': 0.7 })
    return fig