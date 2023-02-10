from functools import reduce

def compose_function(*func): 
    """ Generic function composer making use of functools reduce. 
    
    :param *func: Any number n of input functions to be composed.
    :returns: A new function object.

    Notes:
    Works well in conjunction with functools:partial, where functions can be composed using functions with partially
    filled arguments. This is especially useful for small processing pipelines that are locally defined. e.g:
    threshold_filter_07 = partial(threshold_filter_array, threshold = 0.7)
    extract_top_5 = partial(extract_top_from_array, top_number = 5)
    pipeline = compose(threshold_filter_07, extract_top_5)
    Where pipeline is now a function with one argument (array) based on partial functions with default arguments set.
    """
    def compose(f, g):
        return lambda x : f(g(x))   
    return reduce(compose, func, lambda x : x)