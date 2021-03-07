import numpy as np
import torch
from scipy import stats as stats
from sklearn.metrics import f1_score
from torch import nn as nn

from my_functions import precision_k, print_num_on_tqdm, tqdm_with_num


def training(params, model, train_loader, optimizer):
    device = params["device"]
    batch_total = params["train_batch_total"]
    loss_func = nn.BCELoss()

    model.train()
    losses = []

    # Show loss with tqdm
    with tqdm_with_num(train_loader, batch_total) as loader:
        loader.set_description("Training  ")

        # Batch Loop
        for idx, batch in enumerate(loader):
            # ---------------------- Main Process -----------------------
            data, target = (batch.text.to(device), batch.label.to(device))

            optimizer.zero_grad()

            outputs = model(data)
            outputs = torch.sigmoid(outputs)
            loss = loss_func(outputs, target)

            loss.backward()
            optimizer.step()
            # -----------------------------------------------------------

            # Print training progress
            losses.append(loss.item())

            if idx < batch_total - 1:
                print_num_on_tqdm(loader, loss)
            else:
                loss_epoch = np.mean(losses)
                print_num_on_tqdm(loader, loss_epoch, last=True)


def validating_testing(params, model, data_loader, is_valid=True):
    device = params["device"]
    measure = params["measure"]
    doc_key = is_valid and "valid" or "test"
    batch_total = params[doc_key + "_batch_total"]

    model.eval()

    eval_epoch = 0.0
    target_all = np.empty((0, params["num_of_class"]), dtype=np.int8)
    eval_all = np.empty((0, params["num_of_class"]), dtype=np.float32)

    # Show p@k with tqdm
    with tqdm_with_num(data_loader, batch_total) as loader:
        # Set description to tqdm
        is_valid and loader.set_description("Validating")
        is_valid or loader.set_description("Testing   ")

        with torch.no_grad():
            # Batch Loop
            for idx, batch in enumerate(loader):
                # ---------------------- Main Process -----------------------
                data, target = (batch.text.to(device), batch.label.to("cpu"))
                target = target.detach().numpy().copy()

                outputs = model(data)
                outputs = torch.sigmoid(outputs)
                # -----------------------------------------------------------

                # Print some progress
                outputs = outputs.to("cpu").detach().numpy().copy()
                if "f1" in measure:
                    outputs = outputs >= 0.5

                target_all = np.concatenate([target_all, target])
                eval_all = np.concatenate([eval_all, outputs])

                if idx < batch_total - 1:
                    if "f1" in measure:
                        avg = measure[:-3]
                        eval_batch = f1_score(target, outputs, average=avg)
                    else:
                        k = int(measure[-1])
                        eval_batch = precision_k(target, outputs, k)
                    print_num_on_tqdm(loader, eval_batch, measure)
                else:
                    if "f1" in measure:
                        avg = measure[:-3]
                        eval_epoch = f1_score(target_all, eval_all, average=avg)
                    else:
                        k = int(measure[-1])
                        eval_epoch = precision_k(target_all, eval_all, k)
                    print_num_on_tqdm(loader, eval_epoch, measure, True)

    return eval_epoch
