import torch
import numpy as np

from scipy.sparse import coo_matrix, issparse
from torch.autograd import Variable


class Batcher:

    def __init__(self, data, size):
        self.data = data
        self.size = size
        self.pointer = 0

        if isinstance(self.data, coo_matrix):
            self.data = self.data.tocsr()

    def next_loop(self):
        if self.pointer == splen(self.data):
            self.pointer = 0
        return self.__next__()

    def __next__(self):
        if self.pointer == splen(self.data):
            self.pointer = 0
            raise StopIteration

        next_pointer = min(splen(self.data), self.pointer+self.size)
        to_return = self.data[self.pointer: next_pointer]

        start, end = self.pointer, next_pointer

        self.pointer = next_pointer
        return to_return, splen(to_return), start, end

    def __iter__(self):
        return self


def splen(data):
    try:
        return data.shape[0]
    except:
        return len(data)


def prepare_with_labels(data, labels, label_type="scalar"):
    # Note, we should just be passing in a sparse minibatch here!
    # Doing todense on the entire datset is silly
    if issparse(data):
        data = data.todense()

    v = torch.FloatTensor(np.array(data))
    # if gpu():
    #     return Variable(v.cuda()), Variable(torch.LongTensor(labels).cuda())
    # print(labels)
    # print(Variable(torch.FloatTensor(labels)))
    if label_type == "scalar":
        return Variable(v), Variable(torch.FloatTensor(labels))
    else:
        raise NotImplementedError("Only label type scalar implemented so far")

def prepare(data):
    # Note, we should just be passing in a sparse minibatch here!
    # Doing todense on the entire datset is silly
    if issparse(data):
        data = data.todense()
    v = torch.FloatTensor(np.array(data))
    # if gpu():
    #     return Variable(v.cuda())
    return Variable(v)
