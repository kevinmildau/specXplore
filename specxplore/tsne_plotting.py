import plotly.express as px

def plot_tsne_overview(
    point_selection, selected_class_level, selected_class_data, tsne_df, 
    color_dict):
    if point_selection:  
        selected_point = point_selection["points"][0]["customdata"][0]
        color_dict[selected_class_data[selected_point]] = "#FF10F0"
    tsne_df_tmp = tsne_df
    tsne_df_tmp["clust"] = selected_class_data
    fig = px.scatter(
        tsne_df_tmp, x = "x", y = "y", color = "clust", custom_data=["id"], 
        color_discrete_map=color_dict, render_mode='webgl')
    # for more complex specifications, use go.Scattergl()
    fig.update_layout(clickmode='event+select', 
                        margin = {"autoexpand":True, "b" : 0, "l":0, 
                                "r":50, "t":0})
    fig.update_layout(yaxis_visible=False, yaxis_showticklabels=False, 
        xaxis_visible=False, xaxis_showticklabels=False, 
        title_text='T-SNE Overview', title_x=0.01, title_y=0.01,
        legend_title_text = selected_class_level,
        paper_bgcolor='#feeff4',
        plot_bgcolor='#feeff4', uirevision = False,
        legend = dict(title_font_family='Ubuntu',font=dict(size=6))
        )
    return fig