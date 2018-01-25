from collections import defaultdict, Counter
from scipy.sparse import dok_matrix
from tqdm import tqdm


class Vocab:
    def __init__(self, max_size=None):
        self.vocab = defaultdict(int)
        self.max_size = max_size

        # if max_size is not None:
        #     self.vocab = Counter()

    def add(self, all_items):
        for item in all_items:
            for f in item:
                self.vocab[f] += 1

    def generate_dict(self):
        vocab = dict()
        # if self.max_size is None:
        words = list(self.vocab.keys())
        # else:
        #     words = [a[0] for a in self.vocab.most_common(self.max_size)]
        words.append("UNKNOWN")
        for i, word in enumerate(words):
            vocab[word] = i
        self.vocab = vocab

    def lookup(self, instances):
        rr = []
        for instance in instances:
            ret = defaultdict(int)
            for feature in instance:
                if feature in self.vocab:
                    ret[self.vocab[feature]] += 1
                else:
                    ret[self.vocab["UNKNOWN"]] += 1
            rr.append(ret)
        return rr

    def lookup_sparse(self, data, data_size):
        vocab_size = len(self.vocab)
        dok = dok_matrix((data_size, vocab_size))
        for idx, instance in tqdm(enumerate(data)):
            for feature in instance:
                dim = self.vocab[feature] if feature in self.vocab \
                    else self.vocab["UNKNOWN"]
                dok[idx, dim] += 1
        return dok
