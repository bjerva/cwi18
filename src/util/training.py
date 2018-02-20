import torch


class EarlyStopping:
    def __init__(self, name, patience=8, low_is_good=True, verbose=False):
        self.patience = patience
        self.best_model = None
        self.best_score = None

        self.best_epoch = 0
        self.epoch = 0
        self.low_is_good = low_is_good
        self.name = name
        self.verbose = verbose

    def __call__(self, model, score):
        self.epoch += 1

        if self.best_score is None:
            self.best_score = score

        if self.new_best(score):
            torch.save(model.state_dict(),
                       "models/{0}.best.save".format(self.name))
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
        model.load_state_dict(
            torch.load("models/{0}.best.save".format(self.name)))