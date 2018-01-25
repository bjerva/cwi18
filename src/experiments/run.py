from features.featurize import featurize
from io import get_data
from model import MTMLP
from training import train_model, eval_model

data_en_tr = get_data("en", "train")

feature_functions = []

X_en_tr, y_en_tr = featurize(data_en_tr, feature_functions)

model = MTMLP(X_en_tr.shape[1], [10], [2])

train_model(model, X_en_tr, y_en_tr)
eval_model(model, X_en_tr, y_en_tr)