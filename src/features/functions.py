import os
import pickle

import scipy.sparse
from features.vocab import Vocab
from util.io import *
import kenlm
from util.bpe import infer_spaces
from tqdm import tqdm
import config
import json
from scipy import spatial
import numpy as np
from nltk.corpus import wordnet as wn
from nltk.stem import SnowballStemmer


def load_de2en(path):
    de2en = {}
    for line in open(str(path)):
        line = line.strip()
        if len(line) == 0 or line.startswith("#"):
            continue
        line = line.split("\t")
        if len(line) > 1:
            de2en[line[0]] = line[1]
    return de2en

de2en = load_de2en(config.DE2EN)


class FeatureFunction:

    def __init__(self, name="_abstract", **kwargs):
        super().__init__()
        self.vocab = Vocab()
        self.name = name
        self.kwargs = kwargs

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

    def __repr__(self):
        return self.name

    def toJSON(self):
        return json.dumps(self.name)


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


class Frequency(FeatureFunction):

    def __init__(self, name="frequency", language=None):
        self.lm = self.load_lm(config.LMS[language])
        super().__init__(name)

    @staticmethod
    def load_lm(path):
        return kenlm.LanguageModel(str(path))

    def process(self, data):
        return [self.lm.score(x[TARGET], bos=False, eos=False) for x in data]

    def toJSON(self):
        return self.name


class Synsets(FeatureFunction):
    def __init__(self, name="synsets", language=None):
        lang2wn_lang = {"en": "eng", "es": "spa", "fr": "fra",  "de": "eng"}
        self.lang = language
        self.wn_lang = lang2wn_lang[language]
        super().__init__(name)

    def get_synsets(self, item):
        if self.lang == "de":
            # print(item, de2en.get(item))
            item = de2en.get(item, "")
        return wn.synsets(item, lang=self.wn_lang)

    def process(self, data):
        return [
            # mean synsets length
            np.array([len(self.get_synsets(w)) for w in x[TARGET].split()]).mean()
            for x in data]


class Hypernyms(FeatureFunction):
    def __init__(self, name="hypernyms", language=None):
        lang2wn_lang = {"en": "eng", "es": "spa", "fr": "fra", "de": "eng"}
        self.lang = language
        self.wn_lang = lang2wn_lang[language]
        super().__init__(name)

    def get_hypernym_chain(self, item):
        if self.lang == "de":
            item = de2en.get(item, "")
        hc = []
        synsets = wn.synsets(item, lang=self.wn_lang)
        if synsets:
            synset = synsets[0]
            while synset:
                synset = self.get_hypernym(synset)
                hc.append(synset)
        return hc

    def get_hypernym(self, synset):
        hypernyms = synset.hypernyms()
        if hypernyms:
            return hypernyms[0]
        else:
            return None

    def process(self, data):
        return [
            # mean hypernym chain length
            np.array([len(self.get_hypernym_chain(w)) for w in x[TARGET].split()]).mean()
            for x in data]


class WordList(FeatureFunction):
    def __init__(self, name="word_list", language=None):
        self.word_list = self.load_word_list(config.WORD_LIST[language])
        super().__init__(name)

    def load_word_list(self, path):
        return set([line.strip() for line in open(path)])

    def process(self, data):
        return [
            # fraction of words in target that are in word list
            np.array([w in self.word_list for w in x[TARGET].split()]).mean()
            for x in data]


class CharacterPerplexity(FeatureFunction):
    def __init__(self, name="char_ppl", language=None):
        self.lm = self.load_lm(config.CHAR_LMS[language])
        super().__init__(name)

    @staticmethod
    def load_lm(path):
        return kenlm.LanguageModel(str(path))

    def process(self, data):
        return [min(50, self.lm.perplexity(" ".join(list(x[TARGET]))))
                for x in data]


class StemSurfaceLenghtDist(FeatureFunction):
    def __init__(self, name="stem_surface_len_dist", language=None):
        lang2stemmer_lang = {"en": "english", "de": "german",
                             "fr": "french", "es": "spanish"}
        self.stemmer = SnowballStemmer(lang2stemmer_lang[language])
        super().__init__(name)

    def process(self, data):
        return [
            # maximum difference between surface form and stem for w in target
            max([len(w) - len(self.stemmer.stem(w.lower()))
                 for w in x[TARGET].split()])
            for x in data]


class IsLower(FeatureFunction):
    def __init__(self, name="is_lower", language=None):
        super().__init__(name)

    def process(self, data):
        return [x[TARGET].islower() for x in data]


class WordForm(FeatureFunction):
    def __init__(self, name="word_length"):
        super().__init__(name)

    def process(self, data):
        return [x[TARGET] for x in data]


class WordLength(FeatureFunction):
    def __init__(self, name="word_length", language=None):
        super().__init__(name)

    def process(self, data):
        return [len(x[TARGET]) for x in data]



class NounsCount(FeatureFunction):
    def __init__(self, name="n_nouns", language=None):
        super().__init__(name)

    def process(self, data):
        return [int(x[N_NOUN]) for x in data]


class VerbsCount(FeatureFunction):
    def __init__(self, name="n_verbs", language=None):
        super().__init__(name)

    def process(self, data):
        return [int(x[N_VERB]) for x in data]


class AdjCount(FeatureFunction):
    def __init__(self, name="n_adj", language=None):
        super().__init__(name)

    def process(self, data):
        return [int(x[N_ADJ]) for x in data]


class AdvCount(FeatureFunction):
    def __init__(self, name="n_adv", language=None):
        super().__init__(name)

    def process(self, data):
        return [int(x[N_ADV]) for x in data]


class AdpCount(FeatureFunction):
    def __init__(self, name="n_adp", language=None):
        super().__init__(name)

    def process(self, data):
        return [int(x[N_ADP]) for x in data]


class PropnCount(FeatureFunction):
    def __init__(self, name="n_propn", language=None):
        super().__init__(name)

    def process(self, data):
        return [int(x[N_PROPN]) for x in data]



class NumCount(FeatureFunction):
    def __init__(self, name="n_num", language=None):
        super().__init__(name)

    def process(self, data):
        return [int(x[N_NUM]) for x in data]

class NativeAnnotatorsNumber(FeatureFunction):
    def __init__(self, name="native_annotators_number", language=None):
        super().__init__(name)

    def process(self, data):
        return [int(x[NATIVE_SEEN]) for x in data]


class TargetSentenceSimilarity(FeatureFunction):
    def __init__(self, name="target_sentence_sim", language=None):
        from gensim.models import KeyedVectors
        self.model = KeyedVectors.load_word2vec_format(
            str(config.EMBEDDINGS[language]))
        self.vocabulary = self.model.vocab.keys()
        super().__init__(name)

    def get_avg_embedding(self, item):
        subwords = infer_spaces(item, self.vocabulary)
        subwords_filtered = [sw for sw in subwords if sw in self.model]
        return self.model[subwords_filtered].mean(axis=0).reshape(-1, 1)

    def process(self, data):
        sims = []
        for x in tqdm(data):
            cos = spatial.distance.cosine(self.get_avg_embedding(x[SENTENCE]),
                                          self.get_avg_embedding(x[TARGET]))
            sims.append(1-cos)
        return sims


class PrecomputedTargetSentenceSimilarity(FeatureFunction):
    def __init__(self, name="precomputed_target_sentence_sim", language=None):
        super().__init__(name)

    def process(self, data):
        return [float(x[TARGET_SENT_SIMILARITY]) for x in data]


class Dummy(FeatureFunction):

    def __init__(self, name="dummy"):
        super().__init__(name)

    def process(self, data):
        return [0 for _ in range(len(data))]

