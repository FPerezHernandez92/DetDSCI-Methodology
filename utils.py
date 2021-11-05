import json


def get_useful_objects(path_to_json, scale):
    """ Function to obtain useful objects in the JSON

        Parameters
        ----------
        path_to_json: str
            Path to file "useful_objects.json"
        scale: str
            Scale of the detector. One of 'small'|'large'

        Returns
        -------
        list
            List where each element is a useful object
    """

    with open(path_to_json) as f:
        data_dict = json.load(f)

    key = "useful_objects_" + scale
    useful_objects = data_dict[key]

    return useful_objects
