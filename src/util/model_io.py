import json
from features.functions import FeatureFunction


class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if issubclass(obj, FeatureFunction):
            return obj.__name__
        return json.JSONEncoder.default(self, obj)

