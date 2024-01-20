from dataclasses import dataclass, field

def load_specxplore_object_from_pickle(filepath : str) -> SessionData:
    """ Function loads specXplore object from pickle and checks type validity. """
    with open(filepath, 'rb') as file:
        specxplore_object = pickle.load(file) 
    assert isinstance(specxplore_object, SessionData), (
        'Provided data must be a SessionData object!'
    )
    return specxplore_object


@dataclass
class specxploreSessionData ():
    """ 
    Class interface for specXplore session data containing all infor needed by the specXplore dashboard. 
    """
    # Data Structures
    
    def __post_init__ (self):
        """ Functions applies some basic checks on initiated session data to see whether provided data satisfies
        frontend assumptions. """

        # assert iloc overlap across all features
        # assert each iloc has a corresponding feature_id
        # assert feature_ids unique
        # TODO: determine additional critical asserts.
        return None
    
    
    # constructor
    # getters
    var = ...
    def fun(input):
        ...
