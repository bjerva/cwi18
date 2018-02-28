import random

import torch
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error
from scipy.stats import spearmanr
from util.batching import Batcher, prepare, prepare_with_labels
from sklearn.metrics import f1_score
import numpy as np


def train_model(model, training_datasets, batch_size=64, lr=1e-3, epochs=30,
                dev=None, clip=None, early_stopping=None, l2=1e-5,
                lr_schedule=None, batches_per_epoch=None, shuffle_data=True,
                loss_weights=None, lang_id_weight=0.33):
    """
    Trains a model
    :param model:
    :param training_datasets: list of tuples containing dense matrices
    :param batch_size:
    :param lr:
    :param epochs:
    :param dev:
    :param clip:
    :param early_stopping:
    :param l2:
    :param lr_schedule:
    :param batches_per_epoch:
    :param shuffle_data:
    :param loss_weights:
    :return:
    """
    if loss_weights is None:
        loss_weights = np.ones(len(training_datasets))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    if batches_per_epoch is None:
        batches_per_epoch = sum([len(dataset[0]) for dataset
                                 in training_datasets]) // batch_size
    # print("Batches per epoch:", batches_per_epoch)
    batchers = []

    for training_dataset in training_datasets:
        X, y = training_dataset
        if shuffle_data:
            X, y = shuffle(X, y)
        batcher = Batcher(len(X), batch_size)
        batchers.append(batcher)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_lang_id_loss = []
        epoch_cwi_loss = []
        epoch_data_size = 0
        for b in range(batches_per_epoch):
            task_id = random.choice(range(len(training_datasets)))
            batcher = batchers[task_id]
            X, y = training_datasets[task_id]
            size, start, end = batcher.next_loop()
            d, gold = prepare_with_labels(X[start:end], y[start:end],
                                          model.binary)
            model.train()
            optimizer.zero_grad()
            logits, lang_id_pred = model(d, input_task_id=task_id,
                                         output_all=False, train_mode=True,
                                         output_lang_id=True)
            lang_id_true = np.array([task_id] * size)
            lang_id_true = Variable(torch.LongTensor(lang_id_true)).view(-1)
            lang_id_loss = torch.nn.functional.cross_entropy(lang_id_pred, lang_id_true)
            epoch_lang_id_loss.append(lang_id_loss.data.numpy()[0])

            gold = gold.view([size, 1])
            if model.binary:
                logits = torch.nn.functional.sigmoid(logits)  # don't think we need this as cross_entropy performs softmax
                loss = torch.nn.functional.binary_cross_entropy(logits, gold)
            else:
                loss = (logits - gold).pow(2).mean()
            loss = loss * loss_weights[task_id]
            epoch_cwi_loss.append(loss.data.numpy()[0])
            loss += lang_id_loss * lang_id_weight
            loss.backward()

            epoch_loss += loss.cpu()
            epoch_data_size += size

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)

            optimizer.step()

        # print("Epoch lang id loss:", np.array(epoch_lang_id_loss).mean())
        # print("Epoch CWI loss:", np.array(epoch_cwi_loss).mean())

        if lr_schedule is not None:
            optimizer = lr_schedule(optimizer, epoch)

        if dev is not None:
            X_dev, y_dev = dev
            METRIC_NAME = "F1" if model.binary else "MAE"
            score, corr, _ = eval_model(model, X_dev, y_dev, task_id=task_id,
                                      batch_size=batch_size)
            # print("Epoch Dev {} {:1.4f}".format(METRIC_NAME, score))

            if early_stopping is not None and early_stopping(model, score):
                early_stopping.set_best_state(model)
                break

    if early_stopping is not None:
        early_stopping.set_best_state(model)


def eval_model(model, X, y_true, task_id=0, batch_size=64):
    if model.binary:
        return eval_model_binary(model, X, y_true, task_id=task_id,
                                     batch_size=batch_size)
    else:
        return eval_model_regression(model, X, y_true, task_id=task_id,
                                     batch_size=batch_size)


def eval_model_regression(model, X, y_true, task_id=0, batch_size=64):
    predicted = predict_model(model, X, task_id, batch_size)
    predicted = predicted.reshape([-1])
    mae, rank_corr = 0, float('nan')
    mae = mean_absolute_error(y_true, predicted)
    if predicted.sum() > 0:
        rank_corr = spearmanr(y_true, predicted)[0]
    return mae, rank_corr, predicted


def eval_model_binary(model, X, y_true, task_id=0, batch_size=64):
    predicted = predict_model(model, X, task_id, batch_size)
    predicted = predicted.reshape([-1]) >= 0
    f1 = f1_score(y_true, predicted)
    if predicted.sum() > 0:
        rank_corr = spearmanr(y_true, predicted)[0]
    else:
        rank_corr = float('nan')
    return f1, rank_corr, predicted


def predict_model(model, data, task_id=0, batch_size=64):
    batcher = Batcher(len(data), batch_size)
    predicted = []
    for size, start, end in batcher:
        d = prepare(data[start:end])
        model.eval()
        pred = model(d, input_task_id=task_id, output_all=False,
                     output_lang_id=False).cpu()
        predicted.extend(pred)
    return torch.stack(predicted).data.numpy()


def predict_lang_id(model, data, task_id=0, batch_size=64):
    batcher = Batcher(len(data), batch_size)
    predicted = []
    for size, start, end in batcher:
        d = prepare(data[start:end])
        model.eval()
        _, lang_id_pred = model(d, input_task_id=task_id, output_all=False,
                     output_lang_id=True)
        predicted.extend(lang_id_pred.cpu())
    return torch.stack(predicted).data.numpy()
