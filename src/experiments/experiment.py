from features.featurize import featurize, feature_compatibility
from features.functions import *
from model import MTMLP
from run import train_model, eval_model
from util.io import get_data
import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score, recall_score, \
    precision_score
from util.training import EarlyStopping, split_train_dev
import os
import json
from util.model_io import CustomJSONEncoder

EN = "en"
DE = "de"
ES = "es"
FR = "fr"


def run_experiment(exp_name, train_langs, dev_lang, functions, restarts=1,
                   binary=False, binary_vote_threshold=0.0,
                   hidden_layers=(10, 10), max_epochs=30, batch_size=64,
                   lr=3e-2, dropout=0.2, patience=5,
                   scale_features=True, aux_task_weight=1.0,
                   concatenate_train_data=False, share_input=False):
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
    all_data = {lang: get_data(lang, "Train") + get_data(lang, "Dev")
                for lang in train_langs}

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

    print(feature_functions)

    # Featurize data, each element in the list is a tuple (X, y)
    featurized_data = [
        featurize(all_data[lang], feature_functions[lang], binary=binary,
                  scale_features=scale_features)
        for lang in train_langs
    ]

    dev_lang_index = train_langs.index(dev_lang)
    data_tr, data_dv = split_train_dev(featurized_data, dev_lang_index,
                                       random_splits=False)

    if share_input:
        if not feature_compatibility(functions, train_langs):
            print("Can't concatenate inputs as they differ in their features.")
            return 1

    if concatenate_train_data:
        if not feature_compatibility(functions, train_langs):
            print("Can't concatenate inputs as they differ in their features.")
            return 1
        X = np.concatenate([_x for _x, _y in data_tr])
        y = np.concatenate([_y for _x, _y in data_tr])
        data_tr = [[X, y]]
        dev_lang_index = 0

    X_dv, y_dv = data_dv
    print(X_dv.shape)

    # Perform specified number of random restarts, each restart works as
    # voter in ensemble
    votes = []
    round_performances = []
    metric_name = "F1" if binary else "MAE"

    for round_ in range(1, restarts+1):

        input_dims = [data_tr[task_id][0].shape[1]
                      for task_id in range(len(data_tr))]
        model = MTMLP(input_dims, list(hidden_layers),
                      [1]*len(data_tr), binary=binary, dropout=dropout,
                      share_input=share_input)

        early_stopping = None
        if patience > 0:
            model_prefix = "{}/{}".format(model_dir, round_)
            early_stopping = EarlyStopping(model_prefix, patience,
                                           low_is_good=not binary)

        loss_weights = np.ones(len(data_tr)) * aux_task_weight
        loss_weights[dev_lang_index] = 1
        train_model(model, data_tr, batch_size, lr, max_epochs,
                    early_stopping=early_stopping, dev=data_dv,
                    loss_weights=loss_weights)

        metric, spearman, predictions = eval_model(
            model, X_dv, y_dv, task_id=dev_lang_index)
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
        score = f1_score(y_dv, final_votes)
        recall = recall_score(y_dv, final_votes)
        prec = precision_score(y_dv, final_votes)
        all_scores = "Final P: {:1.4f}, R: {:1.4f}".format(prec, recall)
        print(all_scores)
        exp_log.write(all_scores + "\n")
    else:
        # Take median here
        final_votes = np.median(votes, axis=0)
        score = mean_absolute_error(y_dv, final_votes)
    round_performances = np.array(round_performances)
    final_eval = "Final {} ({}): {:1.4f} (round mean: {:1.4f}, min: {:1.4f}, " \
                 "max: {:1.4f})".format(metric_name, dev_lang, score,
                                        round_performances.mean(),
                                        round_performances.min(),
                                        round_performances.max())
    print(final_eval)
    exp_log.write(final_eval+"\n")
    exp_log.close()

common_funcs = [
    Frequency,
    CharacterPerplexity,
    PrecomputedTargetSentenceSimilarity,
    Synsets,
    Hypernyms,
    NativeAnnotatorsNumber,
    # StemSurfaceLenghtDist
]

funcs = {EN: common_funcs,
         DE: common_funcs,
         ES: common_funcs,
         FR: common_funcs}

run_experiment("test27", [DE,EN,ES], DE, funcs, binary=True,
               restarts=10, max_epochs=1000, lr=1e-3, dropout=0.2,
               binary_vote_threshold=0.0, patience=30, aux_task_weight=.3,
               concatenate_train_data=False,
               hidden_layers=[10,10], share_input=False)

RESTARTS = [5, 10]
PATIENCE = [10, 20]
LR = [3e-2, 1e-3, 3e-3]
DROPOUT = [0.33]
BIN_VOTE_THRESHOLD = [0.0, 0.2, 0.5]
AUX_TASK_WEIGHT = [0.3, 0.5, 1]
CONCAT_TRAIN = [True, False]
SHARE_INPUT = [True, False]
HIDDEN = [[10], [10, 10], [5, 10, 5]]
TRAIN_DATA = [[DE], [DE, EN], [DE, ES], [DE, EN, ES]]
#
# for restarts in RESTARTS:
#   for patience in PATIENCE:
#     for lr in LR:
#       for dropout in DROPOUT:
#         for bvt in BIN_VOTE_THRESHOLD:
#           for aux_weight in AUX_TASK_WEIGHT:
#             for concat in CONCAT_TRAIN:
#               for share in SHARE_INPUT:
#                 for hidden in HIDDEN:
#                   for train_langs in TRAIN_DATA:
#                         exp_name = "rest{}-pat{}-lr{}-dropout{}-bvt{}-aux_" \
#                                  "weight{}-concat{}-share{}-hidden{}-train{}"\
#                             .format(restarts, patience, lr, dropout, bvt,
#                                     aux_weight, concat, share, hidden,
#                                     train_langs)
#
#                         run_experiment(exp_name, train_langs, DE, funcs,
#                                        binary=True, restarts=restarts,
#                                        max_epochs=200, lr=lr, dropout=dropout,
#                                        binary_vote_threshold=bvt,
#                                        patience=patience,target_sentence_sim-Train.txt
#                                        aux_task_weight=aux_weight,
#                                        concatenate_train_data=concat,
#                                        hidden_layers=hidden, share_input=share)
