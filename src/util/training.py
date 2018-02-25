import torch
import random
import numpy as np


def split_train_dev(all_data, dev_lang, random_splits=False, train_rato=0.8):
    train_data = []
    dev_data = []
    for lang_idx in range(len(all_data)):
        if not lang_idx == dev_lang:
            train_data.append(all_data[lang_idx])
        else:
            train_data.append([])
            dev_lang_data = all_data[dev_lang]
            total_size = len(dev_lang_data[0])
            if random_splits:
                train_indices = random.sample(range(total_size),
                                              int(total_size*.8))
            else:
                # get the first 80% for training
                train_indices = list(range(total_size))[:int(total_size *
                                                             train_rato)]
            x_tr, y_tr, x_dv, y_dv = [], [], [], []
            for i in range(total_size):
                x, y = dev_lang_data[0][i], dev_lang_data[1][i]
                if i in train_indices:
                    x_tr.append(x)
                    y_tr.append(y)
                else:
                    x_dv.append(x)
                    y_dv.append(y)
            train_data[lang_idx] = np.array(x_tr), np.array(y_tr)
            dev_data = np.array(x_dv), np.array(y_dv)
    return train_data, dev_data


class EarlyStopping:
    def __init__(self, path_prefix, patience=8, low_is_good=True,
                 verbose=False):
        self.patience = patience
        self.best_model = None
        self.best_score = None

        self.best_epoch = 0
        self.epoch = 0
        self.low_is_good = low_is_good
        self.path_prefix = path_prefix
        self.verbose = verbose

    def __call__(self, model, score):
        self.epoch += 1

        if self.best_score is None:
            self.best_score = score

        if self.new_best(score):
            torch.save(model.state_dict(), self.path_prefix+"_best.model")
            self.best_score = score
            self.best_epoch = self.epoch
            return False

        elif self.epoch > self.best_epoch+self.patience:
            print("Early stopping: Terminate")
            return True
        if self.verbose:
            print("Early stopping: Worse epoch")
        return False

    def new_best(self, score):
        if self.low_is_good:
            return score <= self.best_score
        else:
            return score >= self.best_score

    def set_best_state(self, model):
        print("Loading weights from epoch {0}".format(self.best_epoch))
        model.load_state_dict(torch.load(self.path_prefix+"_best.model"))
