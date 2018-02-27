
import itertools
from multiprocessing import Process
from experiments.experiment import run_experiment, EN, DE, ES, funcs

RESTARTS = [5,10]
PATIENCE = [10]
LR = [3e-2, 1e-3, 3e-3]
DROPOUT = [0.33]
BIN_VOTE_THRESHOLD = [0.0, 0.2, 0.5]
AUX_TASK_WEIGHT = [0.3, 0.5, 1]
CONCAT_TRAIN = [True, False]
SHARE_INPUT = [True, False]
HIDDEN = [[10], [10, 10], [5, 10, 5]]
#TRAIN_DATA = [[DE], [DE, EN], [DE, ES], [DE, EN, ES]]
TEST_LANGS = [EN, DE, ES]


c=0
for test_lang in TEST_LANGS:
    TRAIN_DATA = list(set([(test_lang,)] +
        list(itertools.permutations(TEST_LANGS, 2)) +
        list(itertools.permutations(TEST_LANGS, 3))))
    for restarts in RESTARTS:
        for patience in PATIENCE:
            for lr in LR:
                for dropout in DROPOUT:
                    for bvt in BIN_VOTE_THRESHOLD:
                        for aux_weight in AUX_TASK_WEIGHT:
                            for concat in CONCAT_TRAIN:
                                processes = []
                                for share in SHARE_INPUT:
                                    for hidden in HIDDEN:
                                        done = set()
                                        for train_langs in TRAIN_DATA:
                                            train_langs = list(set(train_langs))
                                            if test_lang in train_langs:
                                                train_langs.remove(test_lang)
                                            train_langs = [test_lang] + train_langs
                                            if tuple(train_langs) in done:
                                                continue
                                            done.add(tuple(train_langs))
                                            exp_name = "rest{}-pat{}-lr{}-dropout{}-bvt{}-aux_weight{}-concat{}-share{}-hidden{}-train{}".format(restarts, patience, lr, dropout, bvt,
                                            aux_weight, concat, share,
                                            '_'.join([str(i) for i in hidden]),
                                            '_'.join(train_langs))
                                            p = Process(target=run_experiment, args=[exp_name, train_langs, test_lang, funcs], kwargs={'binary':True, 'restarts':restarts,
                                            'max_epochs':200, 'lr':lr, 'dropout':dropout,
                                            'binary_vote_threshold':bvt,
                                            'patience':patience,#target_sentence_sim-Train.txt
                                            'aux_task_weight':aux_weight,
                                            'concatenate_train_data':concat,
                                            'hidden_layers':hidden, 'share_input':share})
                                            p.start()
                                            processes.append(p)
                                            '''
                                            run_experiment(exp_name,
                                                   train_langs, test_lang, funcs,
                                                   binary=True, restarts=restarts,
                                                   max_epochs=200, lr=lr, dropout=dropout,
                                                   binary_vote_threshold=bvt,
                                                   patience=patience,#target_sentence_sim-Train.txt
                                                   aux_task_weight=aux_weight,
                                                   concatenate_train_data=concat,
                                                   hidden_layers=hidden, share_input=share)
                                            '''
                                for p in processes:
                                    p.join()

