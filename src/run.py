import random

import torch
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.utils import shuffle
from sklearn.metrics import mean_squared_error
from util.batching import Batcher, prepare, prepare_with_labels


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
    batches = []

    for training_dataset in training_datasets:
        X, y = training_dataset
        if shuffle_data:
            X, y = shuffle(X, y)
        batcher = Batcher(X, batch_size)
        batches.append(batcher)

    for epoch in tqdm(range(epochs)):
        epoch_loss = 0
        epoch_data_size = 0
        for b in range(batches_per_epoch):
            task_id = random.choice(range(len(training_datasets)))
            batcher = batches[task_id]
            # for idx,batcher in enumerate(zip(*batches)):
            X, y = training_datasets[task_id]
            batch, size, start, end = batcher.next_loop()
            # for batch, size, start, end in batcher:
            d, gold = prepare_with_labels(batch, y[start:end],
                                          label_type="scalar")

            model.train()
            optimizer.zero_grad()
            logits_list = model(d)

            logits = logits_list[task_id]
            # print(logits)
            lossfunc = torch.nn.MSELoss()
            loss = lossfunc(logits, gold)
            loss.backward()

            epoch_loss += loss.cpu()
            epoch_data_size += size

            if clip is not None:
                torch.nn.utils.clip_grad_norm(model.parameters(), clip)

            optimizer.step()

        if lr_schedule is not None:
            optimizer = lr_schedule(optimizer, epoch)

        print("Average epoch loss: {0}".format(
            (epoch_loss / epoch_data_size).data.numpy()))

        print("Epoch Train MSE {0}".format(eval_model(
            model, (X, y), task_id=task_id, batch_size=batch_size)))
        if dev is not None:
            X_dev, y_dev = dev
            acc = eval_model(model, (X_dev, y_dev), task_id=task_id,
                             batch_size=batch_size)
            print("Epoch Dev MSE {0}".format(acc))

            if early_stopping is not None and early_stopping(model, acc):
                early_stopping.set_best_state(model)
                break

    if early_stopping is not None:
        early_stopping.set_best_state(model)


def eval_model(model, dataset, task_id=0, batch_size=64, label_type="scalar"):
    data, labels = dataset
    predicted = predict_model(model, data, task_id, batch_size).data.numpy()
    if label_type == "scalar":
        return mean_squared_error(labels, predicted)
    else:
        raise NotImplementedError("Only scalar error evaluation (using MSE)"
                                  "implemented so far.")


def predict_model(model, data, task_id=0, batch_size=64):
    batcher = Batcher(data, batch_size)
    predicted = []
    for batch, size, start, end in batcher:
        d = prepare(batch)
        model.eval()
        logits = model(d)[task_id].cpu()
        predicted.extend(torch.max(logits, 1)[1])
    return torch.stack(predicted)
