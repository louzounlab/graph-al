from src.accelerated_graph_features.graph_converter import convert_graph_to_db_dict


class FeatureWrapper(object):
    """
    This class is a decorator for the pure python function that are exposed to the user of the accelerated feature package.
    The decorator is responsible for doing the tasks that are common to all wrapper functions:
         - Converting the nx.Graph object to a converted graph dictionary.
         - Marking the time for conversion and calculation (if timer is given)
    """

    def __init__(self, func):
        self.f = func

    def __call__(self, graph, **kwargs):
        with_weights = kwargs.get('with_weights', False)
        cast_to_directed = kwargs.get('cast_to_directed', False)

        converted_graph = convert_graph_to_db_dict(graph, with_weights, cast_to_directed)
        if 'timer' in kwargs:
            kwargs['timer'].mark()

        res = self.f(converted_graph, **kwargs)

        if 'timer' in kwargs:
            kwargs['timer'].stop()

        return res
