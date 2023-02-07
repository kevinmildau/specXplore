from specxplore import visuals as visual_utils
import re

def extract_identifiers(plotly_selection_data):
    """ Function extracts custom_data id's from a provided point selection 
    dictionary. 
        
    Args / Parameters
    ------
    plotly_selection_data:
        Selection data as returned by plotly (json format). spec_ids are 
        assumed too be stored inside the 'customdata' component of the 
        selection data at index 0.

    Returns
    ------
    output:
        List of spec_ids corresponding to the selected scatter points.
    
    """
    if plotly_selection_data != None:
        selected_ids = [
            elem["customdata"][0] for elem in plotly_selection_data["points"]]
    else:
        selected_ids = []
    return selected_ids

def update_class(selected_class, class_dict):
    """ Function updates selected_class_data and color_dict elements in line
    with provided selected_class.
        
    Args
    ------
    selected_class:
        String identifying the selected class. The string should be a key in 
        class_dict.
    class_dict:
        A dictionary with class_string keys and corresponding class assignment
        list of each spec_id: {class_string : class_assignment_list}. The class
        assignment list is ordered to correspond to spec_id idx ordering.

    Returns
    ------
    output 0:
        selected_class_data, a list of class assignments for each spec_id 
        in original idx ordering.
    output 1:
        A color dictionary where each unique cluster is a string key with the
        corresponding value being a hex code color: {clust_key : color_string}.
    """
    white_buffer = 20 # --> see visual_utils.construct_grey_palette() 
    selected_class_data = class_dict[selected_class]
    selected_class_data = [
        re.sub('[^A-Za-z0-9]+', '_', elem) for elem in selected_class_data]
    n_colors = len(set(selected_class_data))
    colors = visual_utils.construct_grey_palette(n_colors, white_buffer)
    color_dict = visual_utils.create_color_dict(colors, selected_class_data)
    return selected_class_data, color_dict

def update_expand_level(new_expand_level):
    """ Function updates expand level and placeholder in line with input.

    Expand levels define the number of branching out events for the 
    ego-network represenation. The default value is 1, for all direct 
    connections to the root / ego-node. Large values or values below 1 are
    ill suited for this representation and automatically rejected. 
    A value of 0 would lead to no connections and defeat the point of the
    visualization. Negative values make no sense. Large values will connect
    increasingly irrelevant spectra, or lead to complete graphs depending on
    threshold settings. Sensible defaults are [1 6].
    
    Args / Parameters
    ------
    new_expand_level: 
        New expand level; integer >=1 and should be a small number in line
        with lower and upper limit settings.

    Returns
    ------
    output 0:
        A new expand level setting as integer.
    output 1:
        A new placeholder text for expand level text input box.
    """
    lower_limit = 1
    upper_limit = 6
    if (new_expand_level 
        and new_expand_level >= lower_limit 
        and new_expand_level <= upper_limit
        and isinstance(new_expand_level, int)):
        new_placeholder = (
            f'Expand Level {lower_limit} =< thr <= {upper_limit},'
            f'current: {new_expand_level}')
        return new_expand_level, new_placeholder
    else:
        default_expand_level = 1
        default_placeholder = (
            f'Expand Level {lower_limit} =< thr <= {upper_limit},')
        return default_expand_level,  default_placeholder

def update_threshold(new_threshold):
    """ Function updates threshold level and placeholder in line with input.
    
    Args:
    new_threshold: 
        New threshold; float < 1 and > 0.

    Returns:
    output 0:
        A new threshold setting, float.
    output 1:
        A new placeholder text for threshold text input box.
    """
    if (new_threshold 
        and new_threshold < 1 and new_threshold > 0
        and isinstance(new_threshold, float)):
        new_placeholder = (
            'Threshold 0 < thr < 1,' 
            f'current: {new_threshold}')
        return new_threshold, new_placeholder
    else:
        default_threshold = 0.9
        default_placeholder = "Threshold 0 < thr < 1, def. 0.9"
        return default_threshold,  default_placeholder