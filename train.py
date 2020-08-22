import shutil
from argparse import ArgumentParser

import optuna
import torch
import yaml

from build_problem import BuildProblem
from my_functions import get_num_of_line, out_size


def convert_params(params, length):
    params["hidden_dims"] = 2 ** params["hidden_dims"]
    params["filter_channels"] = 2 ** params["filter_channels"]

    indexes = [i for i in range(params["num_filter_sizes"])]
    del params["num_filter_sizes"]

    params["filter_sizes"] = [params["filter_size_" + str(i)] for i in indexes]
    [params.pop("filter_size_" + str(i)) for i in indexes]

    filter_sizes = params["filter_sizes"]
    params["stride"] = [params["stride_" + str(i)] for i in indexes]
    [params.pop("stride_" + str(i)) for i in indexes]

    args_list = zip(filter_sizes, params["stride"])
    out_sizes = [out_size(length, i, stride=j) for i, j in args_list]
    d_max_list = []
    for i, j in enumerate(out_sizes):
        n_list = [k for k in range(1, j + 1) if j % k < 1]
        n = params["d_max_pool_p_" + str(i)]
        d_max_list.append(n_list[n])

    [params.pop("d_max_pool_p_" + str(i)) for i in indexes]
    params["d_max_pool_p"] = d_max_list

    return params


def main():
    # Config of Args
    parser = ArgumentParser()
    message = "Enable params search."
    action = "store_true"
    parser.add_argument("-s", "--params_search", help=message, action=action)
    message = "Use CPU."
    parser.add_argument("--use_cpu", help=message, action=action)
    args = parser.parse_args()
    is_ps = args.params_search
    use_cpu = args.use_cpu
    use_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_device = use_cpu and torch.device("cpu") or use_device
    term_size = shutil.get_terminal_size().columns

    # Show learning mode
    out_str = is_ps and " Params Search Mode " or " Normal Train Mode "
    print(out_str.center(term_size, "="))

    # Load Params
    with open("params.yml") as f:
        params = yaml.safe_load(f)

    common = params["common"]
    hyper_params = params["hyper_params"]
    normal_train = params["normal_train"]
    params_search = params["params_search"]

    # Show Common Params
    print("\n" + " Params ".center(term_size, "-"))
    print([i for i in sorted(common.items())])
    print("-" * shutil.get_terminal_size().columns)

    # Set of automatically determined params
    params = {}
    params["device"] = use_device
    params["params_search"] = is_ps

    common["cache_path"] += common["cache_path"][-1] == "/" and "" or "/"

    if is_ps:
        num_of_line_train = get_num_of_line(params_search["train_data_path"])
        params["num_of_line_train"] = num_of_line_train
        num_of_line_valid = get_num_of_line(params_search["valid_data_path"])
        params["num_of_line_valid"] = num_of_line_valid

        params["test_data_path"] = None
        params.update(common)
        params.update(params_search)

    else:
        print("\n" + " Hyper Params ".center(term_size, "-"))
        print([i for i in sorted(hyper_params.items())])
        print("-" * shutil.get_terminal_size().columns)

        num_of_line_train = get_num_of_line(normal_train["train_data_path"])
        params["num_of_line_train"] = num_of_line_train
        num_of_line_valid = get_num_of_line(normal_train["valid_data_path"])
        params["num_of_line_valid"] = num_of_line_valid
        num_of_line_test = get_num_of_line(normal_train["test_data_path"])
        params["num_of_line_test"] = num_of_line_test
        params.update(common)
        params.update(hyper_params)
        params.update(normal_train)

    # Build Problem
    trainer = BuildProblem(params)
    trainer.preprocess()

    # Run Training
    is_ps or trainer.run()

    # For Optuna
    if is_ps:
        # Config of Optuna
        optuna.logging.disable_default_handler()
        study = optuna.create_study()

        # Params Search
        study.optimize(trainer.run, n_trials=params["trials"])

        trial = study.best_trial
        params = convert_params(trial.params, params["sequence_length"])

        print("\n\n" + " Best Hyper Params ".center(term_size, "-"))
        print([i for i in sorted(trial.params.items())])
        print("-" * shutil.get_terminal_size().columns)


if __name__ == "__main__":
    main()
