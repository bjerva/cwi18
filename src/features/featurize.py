import numpy as np
from util.io import LABEL_ANY, LABEL_FRACTION
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler


def featurize(dataset, feature_functions, binary=False, scale_features=True,
              return_vectorizer=False):
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
    X = v.fit_transform(per_example_dicts).todense()

    if scale_features:
        mms = MinMaxScaler()
        X = mms.fit_transform(X)

    if binary:
        y = np.array([int(x[LABEL_ANY]) for x in dataset])
    else:
        y = np.array([float(x[LABEL_FRACTION]) for x in dataset])
    if return_vectorizer:
        return X, y, v
    else:
        return X, y
