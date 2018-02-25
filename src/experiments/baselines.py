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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import classification_report

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
    X_tr, y_tr = data_tr[0]
    print(X_tr.shape)

    if binary:
        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X_tr, y_tr)
        pred = clf.predict(X_dv)
        f1 = f1_score(y_dv, pred)
        print("F1: {}".format(f1))
        print(classification_report(y_dv, pred))

    else:
        reg = RandomForestRegressor(n_estimators=100)
        reg.fit(X_tr, y_tr)
        pred = reg.predict(X_dv)
        mae = mean_absolute_error(y_dv, pred)
        print("MAE: ", mae)

    exp_log.close()


common_funcs = [
    Frequency,
    CharacterPerplexity,
    PrecomputedTargetSentenceSimilarity,
    Synsets,
    Hypernyms,
    NativeAnnotatorsNumber,
    StemSurfaceLenghtDist
]

funcs = {EN: common_funcs,
         DE: common_funcs,
         ES: common_funcs,
         FR: common_funcs}

for dvlang in [EN, ES, DE]:
    run_experiment("bl-xx", [dvlang], dvlang, funcs, binary=True,

               concatenate_train_data=False)

