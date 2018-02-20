from features.featurize import featurize
from features.functions import *
from model import MTMLP
from run import train_model, eval_model
from util.io import get_data
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
import math
from util.training import EarlyStopping

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
                   binary_vote_threshold=0.0, hidden_layers=(10, 10),
                   max_epochs=30, batch_size=64, lr=3e-2, patience=5):

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
    round_performances = []
    metric_name = "F1" if binary else "RMSE"

    for round in range(1, restarts+1):
        model = MTMLP(featurized_data_tr[0][0].shape[1], list(hidden_layers),
                      [1, 1], binary=binary)

        early_stopping = None
        if patience > 0:
            early_stopping = EarlyStopping(dev_lang, patience,
                                           low_is_good=not binary)

        train_model(model, featurized_data_tr, batch_size, lr, max_epochs,
                    early_stopping=early_stopping,
                    dev=featurized_data_dv)

        metric, spearman, predictions = eval_model(
            model, X_en_dv, y_en_dv, task_id=0)
        votes.append(predictions)
        print("  Round {} ({}), {}={:1.4f}, rank corr={:1.4f}".format(
            round, dev_lang, metric_name, metric, spearman))
        round_performances.append(metric)
    # Get final votes and compute scores
    votes = np.array(votes)
    if binary:
        # Rule: if any classifier votes yes, take yes
        final_votes = np.mean(votes, axis=0) > binary_vote_threshold
        score = f1_score(y_en_dv, final_votes)
    else:
        # Take median here
        final_votes = np.median(votes, axis=0)
        score = math.sqrt(mean_squared_error(y_en_dv, final_votes))
    round_performances = np.array(round_performances)
    print("Final {} ({}): {:1.4f} "
          "(round mean: {:1.4f}, min: {:1.4f}, max: {:1.4f})".
        format(metric_name, dev_lang, score, round_performances.mean(),
               round_performances.min(), round_performances.max()))


funcs = {EN: [Frequency], DE: [], ES: [], FR: []}
run_experiment([EN], EN, funcs, binary=True, restarts=10, max_epochs=100,
               lr=3e-3, binary_vote_threshold=0.25, patience=20)
