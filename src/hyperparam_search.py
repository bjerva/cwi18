import itertools
from multiprocessing import Process
from experiments.experiment import run_experiment, EN, DE, ES, funcs

RESTARTS = [10]
PATIENCE = [20]
LR = [3e-2, 1e-3, 3e-3]
DROPOUT = [0.33]
BINARY = [True, False]
AUX_TASK_WEIGHT = [0.3, 0.5, 1]
CONCAT_TRAIN = [True, False]
SHARE_INPUT = [True, False]
HIDDEN = [[20], [20,20], [20,30,20]]
#TRAIN_DATA = [[DE], [DE, EN], [DE, ES], [DE, EN, ES]]
TEST_LANGS = [EN, DE, ES]


for restarts in RESTARTS:
    for patience in PATIENCE:
        for lr in LR:
            for dropout in DROPOUT:
                for aux_weight in AUX_TASK_WEIGHT:
                    for concat in CONCAT_TRAIN:
                        for test_lang in TEST_LANGS:
                            TRAIN_DATA = list(set([(test_lang,)] + list(itertools.permutations(TEST_LANGS,2)) +
                                            list(itertools.permutations(TEST_LANGS,3))))
                            for share in SHARE_INPUT:
                                processes = []
                                for hidden in HIDDEN:
                                    for binary in BINARY:
                                        done = set()
                                        for train_langs in TRAIN_DATA:
                                            train_langs = list(set(train_langs))
                                            if test_lang in train_langs:
                                                train_langs.remove(test_lang)
                                            train_langs = [test_lang] + train_langs
                                            if tuple(train_langs) in done:
                                                continue
                                            done.add(tuple(train_langs))
                                            exp_name =  "r2-rest{}-pat{}-lr{}-dropout{}-binary{}-aux_weight{}-concat{}-share{}-hidden{}-train{}".format(restarts,
                                                    patience, lr, dropout,
                                                    binary,
                                            aux_weight, concat, share,
                                            '_'.join([str(i) for i in hidden]),
                                            '_'.join(train_langs))
                                            p = Process(target=run_experiment, args=[exp_name, train_langs, test_lang, funcs], kwargs={'binary':True, 'restarts':restarts,
                                            'max_epochs':200, 'lr':lr, 'dropout':dropout,
                                            'binary_vote_threshold':None,
                                            'binary':binary,
                                            'patience':patience,#target_sentence_sim-Train.txt
                                            'aux_task_weight':aux_weight,
                                            'concatenate_train_data':concat,
                                            'hidden_layers':hidden, 'share_input':share})
                                            p.start()
                                            processes.append(p)
                                
                                for p in processes:
                                    p.join()

