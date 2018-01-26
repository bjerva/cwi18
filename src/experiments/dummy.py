from features.featurize import featurize
from features.functions import Frequency, Dummy
from model import MTMLP
from run import train_model, eval_model
from util.io import get_data

data_en_tr = get_data("english", "Train")
data_de_tr = get_data("german", "Train")

feature_functions = [Dummy("d1"),
                     Dummy("d2")]

X_en_tr, y_en_tr = featurize(data_en_tr, feature_functions)
X_de_tr, y_de_tr = featurize(data_de_tr, feature_functions)

print(X_en_tr.shape, y_en_tr.shape)
print(X_de_tr.shape, y_de_tr.shape)

model = MTMLP(X_en_tr.shape[1], [10], [1, 1])
model.parameters()

train_model(model, [(X_en_tr, y_en_tr), (X_de_tr, y_de_tr)], 64, 0.03, 3)
mse = eval_model(model, (X_en_tr, y_en_tr), task_id=0)
print("Final MSE (en): ", mse)

mse = eval_model(model, (X_de_tr, y_de_tr), task_id=1)
print("Final MSE (de): ", mse)
