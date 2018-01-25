import os
import pickle
import scipy.sparse
from features.vocab import Vocab


class FeatureFunction:

    def __init__(self, name="_abstract"):
        super().__init__()
        self.vocab = Vocab()
        self.name = name

    def inform(self, *datasets):
        print("Inform features")
        for dataset in datasets:
            if dataset is not None:
                generated = self.process(dataset)
                self.vocab.add(generated)
        print("Gen dict")
        self.vocab.generate_dict()

    def lookup(self, data):
        size = len(data)
        processed = self.process(data)
        return self.vocab.lookup_sparse(processed, size)

    def process(self, data):
        raise NotImplementedError("Abstract feature function")


class Features:
    def __init__(self, features=list(), label_name="label",
                 base_path="features"):
        self.feature_functions = features
        self.vocabs = dict()
        self.label_name = label_name
        self.base_path = base_path

    def load(self, *datasets):
        self.inform(*datasets)
        return [self.load_f(dataset, dataset.name) for dataset in datasets]

    def inform(self, *datasets):
        for ff in self.feature_functions:
            ffpath = os.path.join(self.base_path, ff.get_name())

            if not os.path.exists(ffpath):
                os.makedirs(ffpath)

            # If we need train/dev/test data and these don't exist,
            # we have to recreate the features
            if not all([os.path.exists(os.path.join(ffpath,dataset.name)) for dataset in datasets]) or \
                        os.getenv("DEBUG", "").lower() in ["y", "1", "t", "yes"] or \
                        os.getenv("GENERATE", "").lower() in ["y", "1", "t", "yes"]:
                ff.inform(*[dataset.data for dataset in datasets])

    def load_f(self, dataset, name):
        features = []
        for ff in self.feature_functions:
            ffpath = os.path.join(self.base_path, ff.get_name())

            if not os.path.exists(ffpath):
                os.makedirs(ffpath)
            features.append(self.generate_or_load(ff, dataset, name))

        return self.out(features,dataset)

    def out(self, features, ds):
        if ds is not None:
            return scipy.sparse.hstack(features) if len(features) > 1 else features[0], self.labels(ds.data)
        return [[]],[]

    def generate_or_load(self,feature,dataset,name):
        ffpath = os.path.join(self.base_path, feature.get_name())

        if dataset is not None:
            if os.path.exists(os.path.join(ffpath,name)) and not (os.getenv("DEBUG","").lower() in ["y","1","t","yes"]
                    or os.getenv("GENERATE", "").lower() in ["y", "1", "t", "yes"]):
                print("Loading Features for {0}.{1}".format(feature, name))
                with open(os.path.join(ffpath, name), "rb") as f:
                    features = pickle.load(f)

            else:
                print("Generating Features for {0}.{1}".format(feature,name))
                features = feature.lookup(dataset.data)

                with open(os.path.join(ffpath, name), "wb+") as f:
                    pickle.dump(features, f)

            return features

        return None

    def lookup(self,dataset):
        fs = []
        for feature_function in self.feature_functions:
            print("Load {0}".format(feature_function))
            fs.append(feature_function.lookup(dataset.data))
        return self.out(fs,dataset)

    def labels(self,data):
        return [datum[self.label_name] for datum in data]

    def save_vocab(self, mname):
        for ff in self.feature_functions:
            ff.save(mname)

    def load_vocab(self,mname):
        for ff in self.feature_functions:
            ff.load(mname)


# TODO define feature functions
class Frequency(FeatureFunction):

    def __init__(self, name="frequency", language_model=None):
        super().__init__(name)

    def process(self, data):
        # TODO
        pass
