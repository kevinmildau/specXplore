def extract_identifiers(plotly_selection_data):
    """ Function extracts custom_data id's from a provided point selection 
        dictionary. """
    if plotly_selection_data != None:
        selected_ids = [
            elem["customdata"][0] 
            for elem 
            in plotly_selection_data["points"]]
        print("slected _ids:", selected_ids)
    else:
        selected_ids = []
    return selected_ids

def update_class(selected_class, class_dict):
    white_buffer = 20 # --> see visual_utils.construct_grey_palette() 
    selected_class_data = class_dict[selected_class]
    n_colors = len(set(selected_class_data))
    colors = visual_utils.construct_grey_palette(n_colors, white_buffer)
    color_dict = visual_utils.create_color_dict(colors, selected_class_data)
    return selected_class_data, color_dict