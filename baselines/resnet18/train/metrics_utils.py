import numpy as np
import torch

def compute_pr_and_f1(y, y_hat, labels, return_all=False):
    """
    y : torch.tensor[n_data, n_classes] - targets
    y_hat : torch.tensor[n_data, n_classes] - predictions
    labels : List[str] - classes
    return_all : bool - if return all return scores for all the dataset, only mean otherwise
    """
    tps = {}
    fps = {}
    fns = {}
    prec_scores = {}
    recall_scores = {}
    f1_scores = {}

    for i, label in enumerate(labels):
        tp = torch.sum((y_hat[:, i] == 1) & (y[:, i] == 1)).item()
        fp = torch.sum((y_hat[:, i] == 1) & (y[:, i] == 0)).item()
        fn = torch.sum((y_hat[:, i] == 0) & (y[:, i] == 1)).item()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        tps[label] = tp
        fps[label] = fp
        fns[label] = fn
        prec_scores[label] = precision
        recall_scores[label] = recall
        f1_scores[label] = f1

    prec_scores['mean'] = np.mean([v for v in prec_scores.values()])
    prec_scores['std'] = np.std([v for v in recall_scores.values()])

    recall_scores['mean'] = np.mean([v for v in recall_scores.values()])
    recall_scores['std'] = np.std([v for v in recall_scores.values()])

    f1_scores['mean'] = np.mean([v for v in f1_scores.values()])
    f1_scores['std'] = np.std([v for v in f1_scores.values()])

    if return_all:
        return tps, fps, fns, prec_scores, recall_scores, f1_scores
    else:
        return prec_scores['mean'], recall_scores['mean'], f1_scores['mean']


def compute_best_pr_and_f1(y, y_hat, labels, return_all=False):
    """
    y : torch.tensor[n_data, n_classes] - targets
    y_hat : torch.tensor[n_data, n_classes] - predictions
    labels : List[str] - classes
    return_all : bool - if return all return scores for all the dataset, only mean otherwise
    """
    thresholds = np.linspace(0.1, 0.9, 9)  # Test thresholds from 0.1 to 0.9

    best_f1 = -1

    for threshold in thresholds:
        y_hat_thresh = (y_hat >= threshold).int()

        tps, fps, fns, prec_scores, recall_scores, f1_scores = compute_pr_and_f1(y, y_hat_thresh, labels, return_all=True)

        if f1_scores['mean'] > best_f1:
            best_f1 = f1_scores['mean']
            best_tps = tps
            best_fps = fps
            best_fns = fns
            best_threshold = threshold
            best_prec_scores = prec_scores
            best_recall_scores = recall_scores
            best_f1_scores = f1_scores

    best_prec_scores['mean'] = np.mean([v for v in best_prec_scores.values()])
    best_prec_scores['std'] = np.std([v for v in best_prec_scores.values()])

    best_recall_scores['mean'] = np.mean([v for v in best_recall_scores.values()])
    best_recall_scores['std'] = np.std([v for v in best_recall_scores.values()])

    best_f1_scores['mean'] = np.mean([v for v in best_f1_scores.values()])
    best_f1_scores['std'] = np.std([v for v in best_f1_scores.values()])

    if return_all:
        return best_threshold, best_tps, best_fps, best_fns, best_prec_scores, best_recall_scores, best_f1_scores
    else:
        return {'threshold' : best_threshold,
                'mean_best_prec' : best_prec_scores['mean'],
                'mean_best_recall' : best_recall_scores['mean'],
                'mean_best_f1' : best_f1_scores['mean']}