
from multiprocessing import Process
from experiments.experiment import run_experiment

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


for test_lang in TEST_LANGS:
    TRAIN_DATA = [test_lang] + list(itertools.permutations(a, 2)) + list(itertools.permutations(a, 3))
    for restarts in RESTARTS:
        for patience in PATIENCE:
            for lr in LR:
                for dropout in DROPOUT:
                    for bvt in BIN_VOTE_THRESHOLD:
                        for aux_weight in AUX_TASK_WEIGHT:
                            for concat in CONCAT_TRAIN:
                                for share in SHARE_INPUT:
                                    for hidden in HIDDEN:
                                        for train_langs in TRAIN_DATA:
                                            exp_name = "rest{}-pat{}-lr{}-dropout{}-bvt{}-aux_weight{}-concat{}-share{}-hidden{}-train{}".format(restarts, patience, lr, dropout, bvt,
                                            aux_weight, concat, share, hidden,
                                            train_langs)
                                            p1 = Process(target=run_experiment, args=[exp_name, train_langs, test_langs, funcs], kwargs={'binary':True, 'restarts':restarts,
                                            'max_epochs':200, 'lr':lr, 'dropout':dropout,
                                            'binary_vote_threshold':bvt,
                                            'patience':patience,#target_sentence_sim-Train.txt
                                            'aux_task_weight':aux_weight,
                                            'concatenate_train_data':concat,
                                            'hidden_layers':hidden, 'share_input':share})

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
