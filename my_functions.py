import subprocess
import os
import json
import numpy as np
from scipy import io as sio
from sklearn.metrics import average_precision_score
from tqdm import tqdm
from datetime import datetime

def tqdm_with_num(loader, total):
    bar = "{desc}|{bar}| [{remaining}{postfix}]"
    return tqdm(loader, bar_format=bar, total=total)


def print_num_on_tqdm(loader, num, measure=None, last=False):
    out_str = last and "Epoch" or "Batch"

    if measure is None:
        if num < 10.0:
            out_str = " loss={:.8f}/" + out_str
            loader.set_postfix_str(out_str.format(num))
        else:
            num = 9.9999
            out_str = " loss>{:.8f}/" + out_str
            loader.set_postfix_str(out_str.format(num))
    elif "f1" in measure:
        out_str = (measure[:-3] + "={:.8f}/" + out_str).format(num)
        loader.set_postfix_str(out_str)
    else:
        out_str = ("  " + measure + "={:.8f}/" + out_str).format(num)
        loader.set_postfix_str(out_str)


def out_size(l_in, kernel_size, channels, padding=0, dilation=1, stride=1):
    a = l_in + 2 * padding - dilation * (kernel_size - 1) - 1
    b = int(a / stride)
    return (b + 1) * channels


def precision_k(true_mat, score_mat, k):
    rank_mat = np.argsort(score_mat)
    backup = np.copy(score_mat)
    score_mat = np.copy(backup)
    for i in range(rank_mat.shape[0]):
        score_mat[i][rank_mat[i, :-k]] = 0
    score_mat = np.ceil(score_mat)
    mat = np.multiply(score_mat, true_mat)
    num = np.sum(mat, axis=1)
    p = np.mean(num / k).item()
    return p

def print_multiple_metrics(loader, metrics, last=False):
    """Print multiple precision metrics on tqdm progress bar"""
    out_str = last and "Epoch" or "Batch"
    
    metrics_str = f"p@1={metrics['p@1']:.6f}, p@3={metrics['p@3']:.6f}, p@5={metrics['p@5']:.6f}"
    out_str = f" {metrics_str}/{out_str}"
    
    loader.set_postfix_str(out_str)


class MetricsLogger:
    """Logger for training and validation metrics"""
    def __init__(self, log_dir="logs", model_name=None):
        os.makedirs(log_dir, exist_ok=True)
        
        if model_name is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = f"model_{timestamp}"
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.model_name = model_name
        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.train_losses = []
        self.valid_metrics = []
        self.test_metrics = []
        self.best_metrics = {}
        
        print(f"Logging metrics to: {self.model_dir}")
    
    def log_train_loss(self, epoch, batch=None, loss=None, avg_loss=None, extra_metrics=None):
        """Log training loss (batch or epoch level)"""
        if batch is not None:
            log_entry = {
                'epoch': epoch,
                'batch': batch,
                'loss': loss
            }
            if extra_metrics:
                log_entry.update(extra_metrics)
            self.train_losses.append(log_entry)
        else:
            log_entry = {
                'epoch': epoch,
                'avg_loss': avg_loss
            }
            if extra_metrics:
                log_entry.update(extra_metrics)
            self.train_losses.append(log_entry)
        
        self._save_train_logs(self.model_name, self.timestamp)
    
    def log_validation_metrics(self, epoch, metrics):
        """Log validation metrics for an epoch"""
        metrics_data = {
            'epoch': epoch,
            **metrics
        }
        self.valid_metrics.append(metrics_data)
        self._save_validation_logs(self.model_name, self.timestamp)
    
    def log_test_metrics(self, metrics):
        """Log test metrics after training"""
        self.test_metrics = metrics
        self._save_test_logs(self.model_name, self.timestamp)
    
    def log_best_metrics(self, epoch, metrics):
        """Log best model metrics"""
        self.best_metrics = {
            'epoch': epoch,
            **metrics
        }
        self._save_best_logs(self.model_name, self.timestamp)
    
    def _save_train_logs(self, model_name, timestamp):
        """Save training logs to file"""
        with open(os.path.join(self.model_dir, f'{model_name}_train_losses_{timestamp}.json'), 'w') as f:
            json.dump(self.train_losses, f, indent=2)
    
    def _save_validation_logs(self, model_name, timestamp):
        """Save validation logs to file"""
        with open(os.path.join(self.model_dir, f'{model_name}_validation_metrics_{timestamp}.json'), 'w') as f:
            json.dump(self.valid_metrics, f, indent=2)
    
    def _save_test_logs(self, model_name, timestamp):
        """Save test logs to file"""
        with open(os.path.join(self.model_dir, f'{model_name}_test_metrics-{timestamp}.json'), 'w') as f:
            json.dump(self.test_metrics, f, indent=2)
    
    def _save_best_logs(self, model_name, timestamp):
        """Save best model metrics to file"""
        with open(os.path.join(self.model_dir, f'best_model_{model_name}_{timestamp}.json'), 'w') as f:
            json.dump(self.best_metrics, f, indent=2)