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

data_en_tr = get_data("english", "Train")
# data_de_tr = get_data("german", "Train")
# data_dummy_tr = get_data("dummy", "Train")

BINARY = False

feature_functions_common = [
    WordLength()
]
feature_functions_en = feature_functions_common + [
    Frequency(language_model=config.LM_EN),
    # NativeAnnotatorsNumber(),
]

X_en_tr, y_en_tr = featurize(data_en_tr, feature_functions_en, binary=BINARY)
# X_de_tr, y_de_tr = featurize(data_de_tr, feature_functions_en)
X_de_tr, y_de_tr = X_en_tr, y_en_tr

model = MTMLP(X_en_tr.shape[1], [20, 20], [1, 1], binary=BINARY)

train_model(model, [(X_en_tr, y_en_tr), (X_de_tr, y_de_tr)], 32, 0.0003, 10,
            dev=(X_en_tr, y_en_tr))

print("\n====================================\n")
METRIC_NAME = "F1" if BINARY else "RMSE"
metric, spearman, predictions = eval_model(model, X_en_tr, y_en_tr, task_id=0)
print("Final {} (en): {:1.4f}".format(METRIC_NAME, metric))
print("Final rank corr: (en): {:1.4f}".format(spearman))

for i in range(len(X_en_tr)):
    print(predictions[i], y_en_tr[i])
#
# rmse, spearman, predictions = eval_model(model, X_de_tr, y_de_tr, task_id=1)
# print("Final RMSE (de): {:1.4f}".format(rmse))
# print("Final rank corr: (de): {:1.4f}".format(spearman))
