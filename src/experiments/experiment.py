from features.featurize import featurize
from features.functions import *
from model import MTMLP
from run import train_model, eval_model
from util.io import get_data
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
import math

# Don't fix seed, instead do random restarts
# import config
# import torch
# import random
# torch.manual_seed(config.RANDOM_SEED)
# random.seed(config.RANDOM_SEED)
# np.random.seed(config.RANDOM_SEED)

EN = "en"
DE = "de"
ES = "es"
FR = "fr"


def run_experiment(train_langs, dev_lang, functions, restarts=1, binary=False,
                   hidden_layers=(10, 10), epochs=10,
                   batch_size=64, lr=3e-2):

    # Read training data for each training language
    train_data = {lang: get_data(lang, "Train") for lang in train_langs}
    # Read dev data
    dev_data = {dev_lang: get_data(dev_lang, "Dev")}

    # Feature functions shared across languages
    feature_functions_common = [
        WordLength()
    ]

    # Lang-specific feature function instantiations
    feature_functions = {
        lang: feature_functions_common + [function(language=lang)
                                          for function in functions[lang]]
        for lang in train_langs
    }

    # Featurize training data, each element in the list is a tuple (X, y)
    featurized_data_tr = [
        featurize(train_data[lang], feature_functions[lang], binary=binary)
        for lang in train_langs
    ]

    # Featurize training data, this is a single tuple (X, y)
    featurized_data_dv = featurize(
        dev_data[dev_lang], feature_functions[dev_lang], binary=binary)

    # Perform specified number of random restarts, each restart works as
    # voter in ensemble
    X_en_dv, y_en_dv = featurized_data_dv
    votes = []
    for round in range(1, restarts+1):
        model = MTMLP(featurized_data_tr[0][0].shape[1], list(hidden_layers),
                      [1, 1], binary=binary)

        train_model(model, featurized_data_tr, batch_size, lr, epochs,
                    dev=featurized_data_dv)

        print("\n  ====================================\n")
        metric_name = "F1" if binary else "RMSE"
        metric, spearman, predictions = eval_model(
            model, X_en_dv, y_en_dv, task_id=0)
        votes.append(predictions)
        print("  Round {} {} ({}): {:1.4f}".format(
            round, metric_name, dev_lang, metric))
        print("  Round {} rank corr: ({}): {:1.4f}".format(
            round, dev_lang, spearman))

    # Get final votes and compute scores
    votes = np.array(votes)
    print(votes.T[:10])
    if binary:
        final_votes = np.median(votes, axis=0) > 0
        f1 = f1_score(y_en_dv, final_votes)
        print("Final F1 ({}): {:1.4f}".format(dev_lang, f1))
    else:
        final_votes = np.mean(votes, axis=0)
        print(final_votes[:10])
        rmse = math.sqrt(mean_squared_error(y_en_dv, final_votes))
        print("Final RMSE ({}): {:1.4f}".format(dev_lang, rmse))


funcs = {EN: [Frequency], DE: []}
run_experiment([DE], DE, funcs, binary=False, restarts=4, epochs=1, lr=3e-2)
