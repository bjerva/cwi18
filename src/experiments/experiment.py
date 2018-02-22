from features.featurize import featurize
from features.functions import *
from model import MTMLP
from run import train_model, eval_model
from util.io import get_data
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score
from util.training import EarlyStopping
import os
import json
from util.model_io import CustomJSONEncoder

EN = "en"
DE = "de"
ES = "es"
FR = "fr"


def run_experiment(exp_name, train_langs, dev_lang, functions, restarts=1, binary=False,
                   binary_vote_threshold=0.0, hidden_layers=(10, 10),
                   max_epochs=30, batch_size=64, lr=3e-2, patience=5,
                   scale_features=True):
    # Logging
    exp_dir = "../experiments/{}/{}/".format(dev_lang, exp_name)
    model_dir = exp_dir + "models"
    os.mkdir(exp_dir)
    os.mkdir(model_dir)
    args = locals()
    with open(exp_dir + "config.json", "w") as cfg:
        out = json.dumps(args, cls=CustomJSONEncoder)
        cfg.write(out)
    exp_log = open(exp_dir+"log.txt", "w")
    max_len_key = max([len(k) for k in args.keys()])
    for k in sorted(list(args.keys())):
        exp_log.write("{}{}{}\n".format(k, " "*(max_len_key+4-len(k)), args[k]))
    exp_log.write("\n")

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
        featurize(train_data[lang], feature_functions[lang], binary=binary,
                  scale_features=scale_features)
        for lang in train_langs
    ]

    # Featurize training data, this is a single tuple (X, y)
    featurized_data_dv = featurize(
        dev_data[dev_lang], feature_functions[dev_lang], binary=binary,
        scale_features=scale_features)

    # Perform specified number of random restarts, each restart works as
    # voter in ensemble
    X_en_dv, y_en_dv = featurized_data_dv
    votes = []
    round_performances = []
    metric_name = "F1" if binary else "MAE"

    for round_ in range(1, restarts+1):
        model = MTMLP(featurized_data_tr[0][0].shape[1], list(hidden_layers),
                      [1]*len(train_langs), binary=binary)

        early_stopping = None
        if patience > 0:
            model_prefix = "{}/{}".format(model_dir, round_)
            early_stopping = EarlyStopping(model_prefix, patience,
                                           low_is_good=not binary)

        train_model(model, featurized_data_tr, batch_size, lr, max_epochs,
                    early_stopping=early_stopping, dev=featurized_data_dv)

        metric, spearman, predictions = eval_model(
            model, X_en_dv, y_en_dv, task_id=0)
        votes.append(predictions)
        msg = "  Round {} ({}), {}={:1.4f}, rank corr={:1.4f}".format(
            round_, dev_lang, metric_name, metric, spearman)
        print(msg)
        exp_log.write(msg+"\n")
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
        score = mean_absolute_error(y_en_dv, final_votes)
    round_performances = np.array(round_performances)
    final_eval = "Final {} ({}): {:1.4f} (round mean: {:1.4f}, min: {:1.4f}, " \
                 "max: {:1.4f})".format(
        metric_name, dev_lang, score, round_performances.mean(),
        round_performances.min(), round_performances.max())
    print(final_eval)
    exp_log.write(final_eval+"\n")
    exp_log.close()

funcs = {EN: [Frequency, CharacterPerplexity],
         DE: [Frequency, CharacterPerplexity],
         ES: [Frequency, CharacterPerplexity],
         FR: [Frequency, CharacterPerplexity]}

run_experiment("1", [DE, EN, ES], DE, funcs, binary=False, restarts=10,
               max_epochs=100, lr=3e-3, binary_vote_threshold=0.0, patience=10)

run_experiment("2", [DE, EN, ES], DE, funcs, binary=False, restarts=10,
               max_epochs=100, lr=3e-3, binary_vote_threshold=0.0, patience=10)

run_experiment("3", [DE, EN, ES], DE, funcs, binary=False, restarts=10,
               max_epochs=100, lr=3e-3, binary_vote_threshold=0.0, patience=10)

