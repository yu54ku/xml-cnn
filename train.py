import shutil
from argparse import ArgumentParser

import optuna
import torch
import yaml

from build_problem import BuildProblem
from my_functions import out_size
import warnings
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False

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

    filter_channels = params["filter_channels"]
    args_list = zip(filter_sizes, params["stride"])
    out_sizes = [out_size(length, i, filter_channels, stride=j) for i, j in args_list]
    d_max_list = []
    for i, j in enumerate(out_sizes):
        n_list = [k for k in range(1, j + 1) if j % k < 1]
        n = params["d_max_pool_p_" + str(i)]
        d_max_list.append(n_list[n])

    [params.pop("d_max_pool_p_" + str(i)) for i in indexes]
    params["d_max_pool_p"] = d_max_list

    return params


def main():
    parser = ArgumentParser()
    message = "Enable params search."
    action = "store_true"
    parser.add_argument("-s", "--params_search", help=message, action=action)
    message = "Use CPU."
    parser.add_argument("--use_cpu", help=message, action=action)
    message = "Enable Adversarial Training."
    parser.add_argument("-adv", "--adversarial_training", help=message, action=action, default=False)
    message = "only do training."
    parser.add_argument("-t", "--train_only", help=message, action=action, default=False)
    message = "only do testing."
    parser.add_argument("-te", "--test_only", help=message, action=action, default=False)
    message = "use provided model name."
    parser.add_argument("-m", "--model_name", help=message, type=str, default=None)
    args = parser.parse_args()
    is_ps = args.params_search
    use_cpu = args.use_cpu
    use_adv = args.adversarial_training
    use_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_device = use_cpu and torch.device("cpu") or use_device
    term_size = shutil.get_terminal_size().columns

    out_str = is_ps and " Params Search Mode " or " Normal Train Mode "
    print(out_str.center(term_size, "="))

    with open("params.yml") as f:
        params = yaml.safe_load(f)

    common = params["common"]
    hyper_params = params["hyper_params"]
    normal_train = params["normal_train"]
    params_search = params["params_search"]

    if use_adv:
        common["use_adversarial_training"] = True
        common["bit_depth"] = 8
        common["evaluate_adversarial"] = True

    if args.model_name:
        common["model_name"] = args.model_name

    if args.train_only:
        common["only_train"] = True
    
    if args.test_only:
        common["only_test"] = True

    if args.model_name == "rcv1":
        common["vector_cache"] = ".vector_cache/glove.6B.300d.txt"
        common["epsilon"] = 0.1 
    if args.model_name == "eurlex":
        common["vector_cache"] = ".vector_cache/bow_embeddings.txt"
        common["epsilon"] = 0.01

    if args.model_name == "eurlex":
        normal_train["train_data_path"] = "eurlex_data/train.txt"
        normal_train["test_data_path"] = "eurlex_data/test.txt"
        normal_train["valid_data_path"] = "eurlex_data/valid.txt"
        
    print("\n" + " Params ".center(term_size, "-"))
    print([i for i in sorted(common.items())])
    print("-" * shutil.get_terminal_size().columns)

    params = {"device": use_device, "params_search": is_ps}

    common["model_cache_path"] += common["model_cache_path"][-1] == "/" and "" or "/"

    if is_ps:
        params["test_data_path"] = None
        params.update(common)
        params.update(params_search)

    else:
        print("\n" + " Hyper Params ".center(term_size, "-"))
        print([i for i in sorted(hyper_params.items())])
        print("-" * shutil.get_terminal_size().columns)

        params.update(common)
        params.update(hyper_params)
        params.update(normal_train)

    trainer = BuildProblem(params)
    trainer.preprocess()

    is_ps or trainer.run()

    trainer.explainability_data()
    trainer.run_explainability()

    if is_ps:
        optuna.logging.disable_default_handler()
        study = optuna.create_study()

        study.optimize(trainer.run, n_trials=params["trials"])

        trial = study.best_trial
        params = convert_params(trial.params, params["sequence_length"])

        print("\n\n" + " Best Hyper Params ".center(term_size, "-"))
        print([i for i in sorted(params.items())])
        print("-" * shutil.get_terminal_size().columns)


if __name__ == "__main__":
    main()
