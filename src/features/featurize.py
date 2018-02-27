import numpy as np
from util.io import LABEL_ANY, LABEL_FRACTION, LABEL_ANY_ORIG, LABEL_FRACTION_ORIG
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import MinMaxScaler


def featurize(dataset, feature_functions, binary=False, scale_features=True,
              return_vectorizer=False, augmented=True, x_only=False):
    feature2values = {
        func.name: func.process(dataset)
        for func in feature_functions
    }
    # transform to single dict per example
    per_example_dicts = [
        {key: values[i] for key, values in feature2values.items()}
        for i in range(len(dataset))
    ]
    # print(per_example_dicts)
    v = DictVectorizer()
    X = v.fit_transform(per_example_dicts).todense()

    if scale_features:
        mms = MinMaxScaler()
        X = mms.fit_transform(X)

    if x_only:
        return X, None
    else:
        if binary:
            idx = LABEL_ANY if augmented else LABEL_ANY_ORIG
            y = np.array([int(x[idx]) for x in dataset])
        else:
            idx = LABEL_FRACTION if augmented else LABEL_FRACTION_ORIG
            y = np.array([float(x[idx]) for x in dataset])
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
