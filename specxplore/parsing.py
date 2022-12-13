from utils import visuals as visual_utils

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

def update_expand_level(new_expand_level):
    if (new_expand_level 
        and new_expand_level >= 1 
        and new_expand_level <= 6 
        and isinstance(new_expand_level, int)):
        new_placeholder = ('Expand Level 1 =< thr <= 6,'
            f'current: {new_expand_level}')
        return new_expand_level, new_placeholder
    else:
        default_expand_level = 1
        default_placeholder = "Expand Level 1 =< thr <= 6, def. 1"
        return default_expand_level,  default_placeholder

def update_threshold(new_threshold):
    if (new_threshold 
        and new_threshold < 1 
        and new_threshold > 0
        and isinstance(new_threshold, float)):
        new_placeholder = ('Threshold 0 < thr < 1,' 
            f'current: {new_threshold}')
        return new_threshold, new_placeholder
    else:
        default_threshold = 0.9
        default_placeholder = "Threshold 0 < thr < 1, def. 0.9"
        return default_threshold,  default_placeholder