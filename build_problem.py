import math
import shutil

import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torchtext import data

from my_functions import out_size
from utils import training, validating_testing
from xml_cnn import xml_cnn


# Hyper Params used in Params Search
def get_hyper_params(trial, length):
    suggest_int = trial.suggest_int
    suggest_uni = trial.suggest_uniform

    # num_filter_sizes: Num of filter sizes
    # filter_sizes: Size of Conv filters

    # weight_decay: Weight decay for Optimizer
    # hidden_dims: Num of dims of the hidden layer
    # filter_channels: Num of convolution filter channels
    # learning_rate: Learning rate of Optimizer
    # stride: Stride width of Conv filter

    # d_max_list(d_max_pool_p): "p" in Dynamic Max-Pooling

    num_filter_sizes = suggest_int("num_filter_sizes", 3, 4)
    enumerate_f = range(num_filter_sizes)
    filter_sizes = [suggest_int("filter_size_" + str(i), 1, 8) for i in enumerate_f]

    hidden_dims = 2 ** suggest_int("hidden_dims", 5, 10)
    filter_channels = 2 ** suggest_int("filter_channels", 1, 7)
    learning_rate = suggest_uni("learning_rate", 0.000001, 0.01)

    enumerate_f = enumerate(filter_sizes)
    stride = [suggest_int("stride_" + str(i), 1, j) for i, j in enumerate_f]

    # Convert "p" to divisible nums
    # Due to Optuna's specification, only parameters with a linear, logarithmic rate of increase can be set
    args_list = zip(filter_sizes, stride)
    out_sizes = [out_size(length, i, filter_channels, stride=j) for i, j in args_list]
    d_max_list = []
    for i, j in enumerate(out_sizes):
        n_list = [k for k in range(1, j + 1) if j % k < 1]
        n = suggest_int("d_max_pool_p_" + str(i), 0, len(n_list) - 1)
        d_max_list.append(n_list[n])

    params = {
        "stride": stride,
        "hidden_dims": hidden_dims,
        "filter_sizes": filter_sizes,
        "learning_rate": learning_rate,
        "filter_channels": filter_channels,
        "d_max_pool_p": d_max_list,
    }

    return params


def early_stopping(num_of_unchanged, trigger):
    term_size = shutil.get_terminal_size().columns

    if 0 < trigger:
        if trigger - 1 < num_of_unchanged:
            out_str = " Early Stopping "
            print(out_str.center(term_size, "-") + "\n")
            return True

    return False


# Convert labels to vectors
class MakeLabelVector:
    def __init__(self):
        self.uniq_of_cat = []

    def set_label_vector(self, x):
        self.uniq_of_cat += x.split(" ")
        self.uniq_of_cat = list(set(self.uniq_of_cat))
        return x.split(" ")

    def get_label_vector(self, x):
        buf = []
        for i in x:
            buf_2 = [0 for i in range(len(self.uniq_of_cat))]
            for j in i:
                buf_2[self.uniq_of_cat.index(j)] = 1
            buf.append(buf_2)
        return torch.Tensor(buf[:]).float()


class BuildProblem:
    def __init__(self, params):
        self.params = params
        self.train = ""
        self.valid = ""
        self.test = ""
        self.ID = ""
        self.TEXT = ""
        self.LABEL = ""

        self.best_trial_measure = 0.0
        self.num_of_trial = 1

    def preprocess(self):
        print("\nLoading data...  ", end="", flush=True)

        process = MakeLabelVector()
        set_label_vector = process.set_label_vector
        get_label_vector = process.get_label_vector

        # Define fields for torchtext
        length = self.params["sequence_length"]
        self.ID = data.RawField(is_target=False)
        self.LABEL = data.RawField(set_label_vector, get_label_vector, True)
        self.TEXT = data.Field(sequential=True, lower=True, fix_length=length)

        fields = [
            ("id", self.ID),
            ("label", self.LABEL),
            ("text", self.TEXT),
        ]

        datasets = data.TabularDataset.splits(
            path="./",
            train=self.params["train_data_path"],
            validation=self.params["valid_data_path"],
            test=self.params["test_data_path"],
            format="tsv",
            fields=fields,
        )

        if self.params["params_search"]:
            self.train, self.valid = datasets
        else:
            self.train, self.valid, self.test = datasets

        print("Done.", flush=True)

        # Convert words to ID
        print("Converting text to ID...  ", end="", flush=True)
        if self.params["params_search"]:
            self.TEXT.build_vocab(self.train, self.valid)
        else:
            self.TEXT.build_vocab(self.train, self.valid, self.test)

        self.TEXT.vocab.load_vectors("glove.6B.300d")
        print("Done.\n", flush=True)

        # Add parameters that havn't yet been defined
        self.params["uniq_of_cat"] = process.uniq_of_cat
        self.params["num_of_class"] = len(process.uniq_of_cat)

    def run(self, trial=None):
        params = self.params
        is_ps = params["params_search"]
        term_size = shutil.get_terminal_size().columns

        # Show Hyper Params
        if trial is not None:
            sequence_length = params["sequence_length"]
            hyper_params = get_hyper_params(trial, sequence_length)
            self.params.update(hyper_params)
            0 < trial.number and print("\n")
            out_str = " Trial: {} ".format(trial.number + 1)
            print(out_str.center(term_size, "="))
            print("\n" + " Current Hyper Params ".center(term_size, "-"))
            print([i for i in sorted(hyper_params.items())])
            print("-" * shutil.get_terminal_size().columns + "\n")

        # Generate Batch Iterators
        train_loader = data.Iterator(
            self.train,
            batch_size=params["batch_size"],
            device=params["device"],
            train=True,
        )

        valid_loader = data.Iterator(
            self.valid,
            batch_size=params["batch_size"],
            device=params["device"],
            train=False,
            sort=False,
        )

        if not is_ps:
            test_loader = data.Iterator(
                self.test,
                batch_size=params["batch_size"],
                device=params["device"],
                train=False,
                sort=False,
            )

        # Calc Batch Size
        params["train_batch_total"] = math.ceil(
            len(self.train) / params["batch_size"]
        )

        params["valid_batch_total"] = math.ceil(
            len(self.valid) / params["batch_size"]
        )

        if not is_ps:
            params["test_batch_total"] = math.ceil(
                len(self.test) / params["batch_size"]
            )

        # Define xml-cnn model
        model = xml_cnn(params, self.TEXT.vocab.vectors)
        model = model.to(params["device"])
        epochs = params["epochs"]
        learning_rate = params["learning_rate"]

        # Define Optimizer
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if not is_ps:
            ms = [int(epochs * 0.5), int(epochs * 0.75)]
            scheduler = MultiStepLR(optimizer, milestones=ms, gamma=0.1)

        best_epoch = 1
        num_of_unchanged = 1

        measure = params["measure"]
        measure = "f1" in measure and measure[:-3] or measure
        if not is_ps:
            save_best_model_path = params["model_cache_path"] + "best_model.pkl"
        # 学習
        for epoch in range(1, epochs + 1):
            if self.params["params_search"]:
                out_str = " Epoch: {} ".format(epoch)
            else:
                lr = scheduler.get_last_lr()[0]
                term_size = shutil.get_terminal_size().columns
                out_str = " Epoch: {} (lr={:.20f}) ".format(epoch, lr)
            # out_str = " Epoch: {} ".format(epoch)
            print(out_str.center(term_size, "-"))

            # 学習
            training(params, model, train_loader, optimizer)

            # 検証
            val_measure_epoch_i = validating_testing(params, model, valid_loader)

            # 最良モデルの記録と保存
            if epoch < 2:
                best_val_measure = val_measure_epoch_i
                (not is_ps) and torch.save(model, save_best_model_path)
            elif best_val_measure < val_measure_epoch_i:
                best_epoch = epoch
                best_val_measure = val_measure_epoch_i
                num_of_unchanged = 1
                (not is_ps) and torch.save(model, save_best_model_path)
            else:
                num_of_unchanged += 1

            # Show Best Epoch
            out_str = " Best Epoch: {} (" + measure + ": {:.10f}, "
            out_str = out_str.format(best_epoch, best_val_measure)
            if bool(params["early_stopping"]):
                remaining = params["early_stopping"] - num_of_unchanged
                out_str += "ES Remaining: {}) "
                out_str = out_str.format(remaining)
            else:
                out_str += "ES: False) "
            print("\n" + out_str.center(term_size, "-") + "\n")

            # Early Stopping
            if early_stopping(num_of_unchanged, params["early_stopping"]):
                break

            (not is_ps) and scheduler.step()

        if is_ps:
            # Show Best Trials
            if self.best_trial_measure < best_val_measure:
                self.best_trial_measure = best_val_measure
                self.num_of_trial = trial.number + 1
            out_str = " Best Trial: {} (" + measure + ": {:.20f}) "
            out_str = out_str.format(self.num_of_trial, self.best_trial_measure)
            print(out_str.center(term_size, "="))
        else:
            # Testing on Best Epoch Model
            model = torch.load(save_best_model_path)
            test_measure = validating_testing(
                params, model, test_loader, is_valid=False
            )
            out_str = " Finished "
            print("\n\n" + out_str.center(term_size, "=") + "\n")

            out_str = " Best Epoch: {} (" + measure + ": {:.20f}) "
            out_str = out_str.format(best_epoch, test_measure)
            print("\n" + out_str.center(term_size, "-") + "\n")

        return 1 - best_val_measure
