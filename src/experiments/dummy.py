from features.featurize import featurize
from features.functions import *
from model import MTMLP
from run import train_model, eval_model
from util.io import get_data
import config
import torch
import random
import numpy as np

torch.manual_seed(config.RANDOM_SEED)
random.seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

# data_en_tr = get_data("english", "Train")
# data_de_tr = get_data("german", "Train")
data_dummy_tr = get_data("dummy", "Train")

feature_functions_common = [Dummy("d1")]
feature_functions_en = feature_functions_common + [
    Frequency(language_model=config.LM_EN),
    WordLength(),
    NativeAnnotatorsNumber(),
    # WordForm()
]

X_en_tr, y_en_tr = featurize(data_dummy_tr, feature_functions_en)
# X_de_tr, y_de_tr = featurize(data_de_tr, feature_functions_en)
X_de_tr, y_de_tr =X_en_tr, y_en_tr

print(X_en_tr.shape, y_en_tr.shape)
print(X_de_tr.shape, y_de_tr.shape)

model = MTMLP(X_en_tr.shape[1], [10, 10], [1, 1])
model.parameters()

train_model(model, [(X_en_tr, y_en_tr), (X_de_tr, y_de_tr)], 64, 0.003, 100,
            dev=(X_en_tr, y_en_tr))

print("\n====================================\n")
mse, spearman = eval_model(model, X_en_tr, y_en_tr, task_id=0)
print("Final MSE (en): {:1.4f}".format(mse))
print("Final rank corr: (en): {:1.4f}".format(spearman))

mse, spearman = eval_model(model, X_de_tr, y_de_tr, task_id=1)
print("Final MSE (de): {:1.4f}".format(mse))
print("Final rank corr: (de): {:1.4f}".format(spearman))
