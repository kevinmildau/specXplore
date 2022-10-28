import plotly.express as px

def plot_tsne_overview(TSNE_DF, color_dict, selected_class_data, selected_class_level):
    # Construct T-SNE plot
    tsne_df_tmp = TSNE_DF

    tsne_df_tmp["clust"] = selected_class_data
    fig = px.scatter(tsne_df_tmp, x = "x", y = "y", color = "clust", 
                    custom_data=["id"], color_discrete_map=color_dict,
                    render_mode='webgl') # <-- using efficient and highly scalable render mode; a million points easily covered.
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