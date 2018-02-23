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


def feature_compatibility(functions, train_langs):
    relevant_functions = [functions[lang] for lang in train_langs]
    print(relevant_functions)
    for i in range(len(train_langs)):
        if not len(relevant_functions[i-1]) == len(relevant_functions[i]):
            print("Fetaure functions don't have the same lengths!")
            return False
    for j in range(len(relevant_functions[0])):
        for i in range(len(train_langs)):
            if not relevant_functions[i][j].__name__ \
                    == relevant_functions[i-1][j].__name__:
                print("Feature functions at same positions don't have the "
                      "same names across languages")
                return False
    return True
