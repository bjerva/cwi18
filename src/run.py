import random

import torch
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr
from util.batching import Batcher, prepare, prepare_with_labels, splen
import math
from sklearn.metrics import f1_score


def train_model(model, training_datasets, batch_size=64, lr=1e-3, epochs=30,
                dev=None, clip=None, early_stopping=None, l2=1e-5,
                lr_schedule=None, batches_per_epoch=None, shuffle_data=True):
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
    :return:
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    if batches_per_epoch is None:
        batches_per_epoch = sum([len(dataset[0]) for dataset
                                 in training_datasets]) // batch_size
    print("Batches per epoch:", batches_per_epoch)
    batchers = []

    for training_dataset in training_datasets:
        X, y = training_dataset
        if shuffle_data:
            X, y = shuffle(X, y)
        batcher = Batcher(len(X), batch_size)
        batchers.append(batcher)

    X, y = None, None
    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
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
            logits_all_tasks = model(d)
            logits = logits_all_tasks[task_id]
            gold = gold.view([size, 1])
            if model.binary:
                logits = torch.nn.functional.sigmoid(logits)
                loss = torch.nn.functional.binary_cross_entropy(logits, gold)
            else:
                loss = (logits - gold).pow(2).sum()
            loss.backward()

            epoch_loss += loss.cpu()
            epoch_data_size += size

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)

            optimizer.step()

        if lr_schedule is not None:
            optimizer = lr_schedule(optimizer, epoch)

        # print("Average epoch loss: {0}".format(
        #     (epoch_loss / epoch_data_size).data.numpy()))
        #
        # print("Latest Epoch Train RMSE {0}".format(eval_model(
        #     model, X, y, task_id=task_id, batch_size=batch_size)[0]))
        if dev is not None:
            X_dev, y_dev = dev
            METRIC_NAME = "F1" if model.binary else "RMSE"
            metric, corr, _ = eval_model(model, X_dev, y_dev, task_id=task_id,
                                      batch_size=batch_size)
            print("Epoch Dev {} {:1.4f}".format(METRIC_NAME, metric))
            # print("Epoch Dev Corr {:1.4f}".format(corr))

            if early_stopping is not None and early_stopping(model, metric):
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
    predicted = predict_model(model, X, task_id, batch_size).data.numpy()
    predicted = predicted.reshape([-1])
    rmse, rank_corr = 0, float('nan')
    rmse = math.sqrt(mean_squared_error(y_true, predicted))
    if predicted.sum() > 0:
        rank_corr = spearmanr(y_true, predicted)[0]
    return rmse, rank_corr, predicted


def eval_model_binary(model, X, y_true, task_id=0, batch_size=64):
    predicted = predict_model(model, X, task_id, batch_size).data.numpy()
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
        pred = model(d)[task_id].cpu()
        predicted.extend(pred)
    return torch.stack(predicted)
