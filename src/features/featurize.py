import numpy as np
from util.io import LABEL_FRACTION
from sklearn.feature_extraction import DictVectorizer


def featurize(dataset, feature_functions, return_vectorizer=False):
    feature2values = {
        func.name: func.process(dataset)
        for func in feature_functions
    }
    print(len(dataset))
    # transform to single dict per example
    per_example_dicts = [
        {key: values[i] for key, values in feature2values.items()}
        for i in range(len(dataset))
    ]
    v = DictVectorizer()
    X = v.fit_transform(per_example_dicts)
    y = np.array([x[LABEL_FRACTION] for x in dataset])
    if return_vectorizer:
        return X, y, v
    else:
        return X, y
