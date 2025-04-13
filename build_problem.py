import math
import shutil
import numpy as np
import torch
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from custom_data import XMLCNNDataset, Iterator
import matplotlib.pyplot as plt
from utils import training, validating_testing, adversarial_training, adversarial_validating_testing,explainability_data_extract
from my_functions import out_size, MetricsLogger
import os
import pickle
from xml_cnn import xml_cnn
from adversarial_defense import FGSM
from lime.lime_text import LimeTextExplainer
import shap
from collections import defaultdict
from transformers import AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

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

        ## -----------------custom dataloader implementaion ------------
        self.train_dataset = XMLCNNDataset(
            self.params["train_data_path"],
            max_length=self.params["sequence_length"]
        )
        
        valid_dataset = XMLCNNDataset(
            self.params["valid_data_path"],
            vocab=self.train_dataset.vocab,
            build_vocab=False,
            max_length=self.params["sequence_length"],
            label_list=self.train_dataset.label_list,
            label_to_idx=self.train_dataset.label_to_idx
        )
        
        if not self.params["params_search"]:
            test_dataset = XMLCNNDataset(
                self.params["test_data_path"],
                vocab=self.train_dataset.vocab,
                build_vocab=False,
                max_length=self.params["sequence_length"],
                label_list=self.train_dataset.label_list,
                label_to_idx=self.train_dataset.label_to_idx
            )
        
        self.train = self.train_dataset
        self.valid = valid_dataset
        if not self.params["params_search"]:
            self.test = test_dataset
        
        # embeddings = train_dataset.load_vectors('.vector_cache/glove.6B.300d.txt')
        
        vector_path = self.params.get("vector_cache", ".vector_cache/glove.6B.300d.txt")
        embeddings = self.train_dataset.load_vectors(vector_path)

        class DummyText:
            def __init__(self):
                self.vocab = DummyVocab()
                
        class DummyVocab:
            def __init__(self):
                self.vectors = embeddings
                
        self.TEXT = DummyText()
        
        self.params["uniq_of_cat"] = self.train_dataset.label_list
        self.params["num_of_class"] = len(self.train_dataset.label_list)
        
        print("Done.", flush=True)

    def run(self, trial=None):
        params = self.params
        is_ps = params["params_search"]
        term_size = shutil.get_terminal_size().columns
        use_adversarial = params.get("use_adversarial_training", False)

        only_test = params.get("only_test", False)
        only_train = params.get("only_train", False)

        model_name = params.get("model_name", "xml_cnn")
        if model_name is None:
            model_name = "xml_cnn"
        if use_adversarial:
            model_name += "_adv"

        save_best_model_path = params["model_cache_path"] + f"best_model_{model_name}.pkl"
        
        if only_test and not only_train and not is_ps:
            model_name = params.get("model_name", "xml_cnn")
            if model_name is None:
                model_name = "xml_cnn"
            # if use_adversarial:
            #     model_name += "_adv"
                
            test_loader = Iterator(
                self.test,
                batch_size=params["batch_size"],
                device=params["device"],
                train=False
            )
            
            params["test_batch_total"] = math.ceil(
                len(self.test) / params["batch_size"]
            )
            
            logger = None
            if params.get("enable_logging", False):
                log_dir = params.get("log_dir", "logs")
                model_name_log = params.get("model_name", None)
                logger = MetricsLogger(log_dir, model_name_log)
            
            save_best_model_path = params["model_cache_path"] + f"best_model_{model_name}.pkl"
            print(save_best_model_path)
            print("\n" + " Test Only Mode " .center(term_size, "="))
            
            if os.path.exists(save_best_model_path):
                model = torch.load(save_best_model_path)
                model = model.to(params["device"]) 

                if params.get("evaluate_adversarial", False) and use_adversarial:
                    test_measure = adversarial_validating_testing(
                        params, model, test_loader, 0, is_valid=False, logger=logger
                    )
                else:
                    test_measure = validating_testing(
                        params, model, test_loader, 0, is_valid=False, logger=logger
                    )
                    
                out_str = " Finished Testing "
                print("\n\n" + out_str.center(term_size, "=") + "\n")
                
                return 0
            else:
                print(f"\nError: Model file {save_best_model_path} not found!")
                return 0
        
        logger = None
        if params.get("enable_logging", False) and not is_ps:
            log_dir = params.get("log_dir", "logs")
            model_name = params.get("model_name", None)
            logger = MetricsLogger(log_dir, model_name)

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

        # Generate Batch Iterators (custom iterator)
        train_loader = Iterator(
            self.train,
            batch_size=params["batch_size"],
            device=params["device"],
            train=True
        )

        valid_loader = Iterator(
            self.valid,
            batch_size=params["batch_size"],
            device=params["device"],
            train=False
        )

        if not is_ps:
            test_loader = Iterator(
                self.test,
                batch_size=params["batch_size"],
                device=params["device"],
                train=False
            )

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

        model = xml_cnn(params, self.TEXT.vocab.vectors)
        model = model.to(params["device"])
        epochs = params["epochs"]
        learning_rate = params["learning_rate"]

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if not is_ps:
            ms = [int(epochs * 0.5), int(epochs * 0.75)]
            scheduler = MultiStepLR(optimizer, milestones=ms, gamma=0.1)

        best_epoch = 1
        num_of_unchanged = 1

        measure = params["measure"]
        measure = "f1" in measure and measure[:-3] or measure

        model_name = params["model_name"]
        if model_name is None:
            model_name = "xml_cnn"
        if use_adversarial:
            model_name += "_adv"

        if not is_ps:
            save_best_model_path = params["model_cache_path"] + f"best_model_{model_name}.pkl"
        
        for epoch in range(1, epochs + 1):
            if self.params["params_search"]:
                out_str = " Epoch: {} ".format(epoch)
            else:
                lr = scheduler.get_last_lr()[0]
                term_size = shutil.get_terminal_size().columns
                out_str = " Epoch: {} (lr={:.20f}) ".format(epoch, lr)
            print(out_str.center(term_size, "-"))
            
            if use_adversarial:
                adversarial_training(params, model, train_loader, optimizer, epoch, logger)
            else:
                training(params, model, train_loader, optimizer, epoch, logger)

            if params.get("evaluate_adversarial", False) and use_adversarial:
                val_measure_epoch_i = adversarial_validating_testing(params, model, valid_loader, epoch, True, logger)
            else:
                val_measure_epoch_i = validating_testing(params, model, valid_loader, epoch, True, logger)

            
            if epoch < 2:
                best_val_measure = val_measure_epoch_i
                (not is_ps) and torch.save(model, save_best_model_path)
                if logger is not None:
                    if "f1" in measure:
                        logger.log_best_metrics(epoch, {measure: best_val_measure})
                    else:
                        # For precision metrics - use the metrics we already have
                        # The p@1, p@3, p@5 values are already computed in validating_testing
                        # and logged there, so here we just log the best measure
                        logger.log_best_metrics(epoch, {
                            measure: best_val_measure  # Just log the primary metric
                        })
            elif best_val_measure < val_measure_epoch_i:
                best_epoch = epoch
                best_val_measure = val_measure_epoch_i
                num_of_unchanged = 1
                (not is_ps) and torch.save(model, save_best_model_path)

                # Log best metrics
                if logger is not None:
                    if "f1" in measure:
                        logger.log_best_metrics(epoch, {measure: best_val_measure})
                    else:
                        logger.log_best_metrics(epoch, {
                            measure: best_val_measure  
                        })
            else:
                num_of_unchanged += 1

            out_str = " Best Epoch: {} (" + measure + ": {:.10f}, "
            out_str = out_str.format(best_epoch, best_val_measure)
            if bool(params["early_stopping"]):
                remaining = params["early_stopping"] - num_of_unchanged
                out_str += "ES Remaining: {}) "
                out_str = out_str.format(remaining)
            else:
                out_str += "ES: False) "
            print("\n" + out_str.center(term_size, "-") + "\n")

            if early_stopping(num_of_unchanged, params["early_stopping"]):
                break

            (not is_ps) and scheduler.step()

        if is_ps:
            if self.best_trial_measure < best_val_measure:
                self.best_trial_measure = best_val_measure
                self.num_of_trial = trial.number + 1
            out_str = " Best Trial: {} (" + measure + ": {:.20f}) "
            out_str = out_str.format(self.num_of_trial, self.best_trial_measure)
            print(out_str.center(term_size, "="))
        else:
            only_train = params.get("only_train", False)
            only_test = params.get("only_test", False)

            if not only_train or only_test:
                model = torch.load(save_best_model_path)
                
                if params.get("evaluate_adversarial", False) and use_adversarial:
                    test_measure = adversarial_validating_testing(
                        params, model, test_loader, best_epoch, is_valid=False, logger=logger
                    )
                else:
                    test_measure = validating_testing(
                        params, model, test_loader, best_epoch, is_valid=False, logger=logger
                    )
                    
                out_str = " Finished "
                print("\n\n" + out_str.center(term_size, "=") + "\n")

                out_str = " Best Epoch: {} (" + measure + ": {:.20f}) "
                out_str = out_str.format(best_epoch, test_measure)
                print("\n" + out_str.center(term_size, "-") + "\n")

        return 1 - best_val_measure


    def explainability_data(self):
        params = self.params
        is_ps = params["params_search"]
        term_size = shutil.get_terminal_size().columns
        use_adversarial = params.get("use_adversarial_training", False)

        only_test = params.get("only_test", False)
        only_train = params.get("only_train", False)

        model_name = params.get("model_name", "xml_cnn")
        if model_name is None:
            model_name = "xml_cnn"
        if use_adversarial:
            model_name += "_adv"

        save_best_model_path = params["model_cache_path"] + f"best_model_{model_name}.pkl"

        test_loader = Iterator(
            self.test,
            batch_size=params["batch_size"],
            device=params["device"],
            train=False
        )
        
        params["test_batch_total"] = math.ceil(len(self.test) / params["batch_size"])

        logger = None
        if params.get("enable_logging", False):
            log_dir = params.get("log_dir", "logs")
            model_name_log = params.get("model_name", None)
            logger = MetricsLogger(log_dir, model_name_log)

        print("\n" + " Test Only Mode ".center(term_size, "="))

        if os.path.exists(save_best_model_path):
            model = torch.load(save_best_model_path, weights_only=False)

            print("Explaining on test data...")
            explanation_data = explainability_data_extract(params, model, test_loader)

            output_path = os.path.join(params["model_cache_path"], "lime_test_data_adv.pkl")
            with open(output_path, "wb") as f:
                pickle.dump(explanation_data, f)

            print(f"Saved explanation data to {output_path}")

    def load_explanation_data(self, pkl_path):
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    def load_test_text(self, test_txt_path):
        with open(test_txt_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f]

    def map_ids_to_raw_text(self, pkl_data, raw_text_lines):
        mapped_data = []
        for idx, probs in pkl_data:
            try:
                idx = int(idx)
                if 0 <= idx < len(raw_text_lines):
                    raw_line = raw_text_lines[idx]
                    text = raw_line.strip()
                    if text and text.split()[0].isdigit():
                        text = ' '.join(text.split()[1:])
                    mapped_data.append((text, probs))
            except Exception as e:
                print(f"Error processing idx={idx}: {e}")
        return mapped_data

    def encode_texts(self, text_list, vocab, max_length):
        encoded = []
        for text in text_list:
            tokens = text.split()
            indices = [vocab.get(token, vocab['<unk>']) for token in tokens]
            if len(indices) < max_length:
                indices += [vocab['<pad>']] * (max_length - len(indices))
            else:
                indices = indices[:max_length]
            encoded.append(indices)
        return torch.tensor(encoded, dtype=torch.long)

    def model_predict_fn(self, text_list, model, vocab, max_length):
        model = model.cpu()
        
        batch_size = 8
        all_probs = []
        
        for i in range(0, len(text_list), batch_size):
            batch_texts = text_list[i:i+batch_size]
            input_tensor = self.encode_texts(batch_texts, vocab, max_length)
            
            with torch.no_grad():
                logits = model(input_tensor)
                probs = torch.sigmoid(logits).numpy()
                all_probs.append(probs)
                
        return np.vstack(all_probs) if len(all_probs) > 1 else all_probs[0]

    def run_lime_on_example(self, example_text, model_predict_fn_wrapper, class_names, num_features=10, idx=None):
        explainer = LimeTextExplainer(class_names=class_names)
        explanation = explainer.explain_instance(
            example_text,
            model_predict_fn_wrapper,
            num_features=num_features,
            labels=np.where(model_predict_fn_wrapper([example_text])[0] > 0.5)[0].tolist()
        )
        if idx is not None:
            explanation.save_to_file(f"lime_results/lime_example_{idx}.html")
        return explanation

    def run_shap_on_examples(self, texts, model_predict_fn_wrapper, class_names, num_samples=5):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(model_predict_fn_wrapper, masker)

        texts_to_explain = texts[:num_samples]
        shap_values = explainer(texts_to_explain)

        output_dir = 'shap_results'
        os.makedirs(output_dir, exist_ok=True)

        for i, text in enumerate(texts_to_explain):
            preds = model_predict_fn_wrapper([text])[0]
            predicted_labels = np.where(preds > 0.5)[0]

            print(f"\nSHAP Explanation for Example #{i}:")
            for label_idx in predicted_labels:
                print(f"\nClass: {class_names[label_idx]}")
                if shap_values[i].values.ndim == 2:
                    values = shap_values[i].values[:, label_idx]
                else:
                    values = shap_values[i].values
                token_value_pairs = list(zip(shap_values[i].data, values))
                sorted_pairs = sorted(token_value_pairs, key=lambda x: abs(x[1]), reverse=True)[:10]
                for word, value in sorted_pairs:
                    print(f"{word}: {value:.4f}")

            explanation_text = shap.plots.text(shap_values[i], display=False)
            html_path = os.path.join(output_dir, f"shap_example_{i}.html")
            with open(html_path, "w") as f:
                f.write(explanation_text)
            print(f" Saved: {html_path}")

    def run_shap_global_insight(self, texts, model_predict_fn_wrapper, num_samples=100):
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        masker = shap.maskers.Text(tokenizer)
        explainer = shap.Explainer(model_predict_fn_wrapper, masker)

        texts_to_explain = texts[:num_samples]
        shap_values = explainer(texts_to_explain)

        token_impact = defaultdict(float)

        for explanation in shap_values:
            for token, value in zip(explanation.data, explanation.values):
                token_impact[token] += np.sum(np.abs(value))

        sorted_tokens = sorted(token_impact.items(), key=lambda x: x[1], reverse=True)
        top_tokens = sorted_tokens[:20]

        tokens, impacts = zip(*top_tokens)
        plt.figure(figsize=(12, 6))
        plt.barh(tokens[::-1], impacts[::-1], color='orchid')
        plt.xlabel("Cumulative SHAP Value (Absolute)")
        plt.title("Top 20 Global Influential Tokens")
        plt.tight_layout()
        plt.savefig("shap_global_tokens.png")
        plt.show()

    def run_explainability(self):
        term_size = shutil.get_terminal_size().columns
        use_adversarial = self.params.get("use_adversarial_training", False)

        model_name = self.params.get("model_name", "xml_cnn")
        if use_adversarial:
            model_name += "_adv"
        save_best_model_path = os.path.join(self.params["model_cache_path"], f"best_model_{model_name}_adv.pkl")

        print("\n" + " Test Only Mode ".center(term_size, "="))

        if os.path.exists(save_best_model_path):
            model = torch.load(save_best_model_path, map_location=torch.device("cpu"))
            model.eval()

        def model_predict_fn_wrapper(texts):
            return self.model_predict_fn(texts, model, self.train_dataset.vocab, self.train_dataset.max_length)

        try:
            explanation_data = self.load_explanation_data(".model_cache/lime_test_data_adv.pkl")
        except:
            explanation_data = self.load_explanation_data(os.path.join(self.params["model_cache_path"], "lime_test_data.pkl"))
        
        raw_lines = self.load_test_text("data/test.txt")
        lime_ready_data = self.map_ids_to_raw_text(explanation_data, raw_lines)

        text, prob = lime_ready_data[0]

        class_names = self.train_dataset.label_list

        os.makedirs("lime_results", exist_ok=True)

        max_examples = 3
        for i, (text, prob) in enumerate(lime_ready_data[:max_examples]):
            print(f"Running LIME for example #{i}...")
            try:
                lime_expl = self.run_lime_on_example(text, model_predict_fn_wrapper, class_names, 
                                                num_features=5, idx=i)
                
                labels = np.where(model_predict_fn_wrapper([text])[0] > 0.5)[0].tolist()
                for label in labels:
                    lime_weights = lime_expl.as_list(label=label)
                    print(f"\nLIME Top Features for class {class_names[label]}:")
                    for word, weight in lime_weights:
                        print(f"{word}: {weight:.4f}")
                        
            except Exception as e:
                print(f"Error processing example #{i}: {str(e)}")
                continue

        texts = [text for text, prob in lime_ready_data]
        self.run_shap_on_examples(texts, model_predict_fn_wrapper, self.train_dataset.label_list)
        self.run_shap_global_insight(texts, model_predict_fn_wrapper)