from features.featurize import featurize
from features.functions import Frequency, Dummy
from model import MTMLP
from training import train_model, eval_model
from util.io import get_data

data_en_tr = get_data("english", "Train")

feature_functions = [Dummy()]

X_en_tr, y_en_tr = featurize(data_en_tr, feature_functions)

print(X_en_tr.shape, y_en_tr.shape)

model = MTMLP(X_en_tr.shape[1], [10], [2])

train_model(model, X_en_tr, y_en_tr)
eval_model(model, X_en_tr, y_en_tr)