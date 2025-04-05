import numpy as np
import torch

def compute_pr_and_f1(y, y_hat, labels, return_all=False): # TODO rename return_all
    """
    # TODO
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
        return {'tps' : tps,
                'fps' : fps,
                'fns' : fns,
                'prec_scores' : prec_scores,
                'recall_scores' : recall_scores,
                'f1_scores' : f1_scores}
    else:
        return {'mean_prec_scores' : prec_scores['mean'],
                'mean_recall_scores' : recall_scores['mean'],
                'mean_f1_scores' : f1_scores['mean']}


def compute_best_pr_and_f1(y, y_hat, labels, return_all=False):
    """
    # TODO
    """
    thresholds = np.linspace(0.01, 0.99, 99)

    best_thresholds = {}
    best_prec_scores = {}
    best_recall_scores = {}
    best_f1_scores = {}
    best_f1s = {label: -1 for label in labels}

    for threshold in thresholds:
        y_hat_thresh = (y_hat >= threshold).int()

        prs_f1s = compute_pr_and_f1(y, y_hat_thresh, labels, return_all=True)

        for label in labels:
            if prs_f1s['f1_scores'][label] > best_f1s[label]:
                best_f1_scores[label] = prs_f1s['f1_scores'][label]
                best_thresholds[label] = threshold
                best_prec_scores[label] = prs_f1s['prec_scores'][label]
                best_recall_scores[label] = prs_f1s['recall_scores'][label]
                best_f1_scores[label] = prs_f1s['f1_scores'][label]

    best_prec_scores['mean'] = np.mean([v for v in best_prec_scores.values()])
    best_prec_scores['std'] = np.std([v for v in best_prec_scores.values()])

    best_recall_scores['mean'] = np.mean([v for v in best_recall_scores.values()])
    best_recall_scores['std'] = np.std([v for v in best_recall_scores.values()])

    best_f1_scores['mean'] = np.mean([v for v in best_f1_scores.values()])
    best_f1_scores['std'] = np.std([v for v in best_f1_scores.values()])

    if return_all:
        return {'best_thresholds': best_thresholds,
                'best_precisions': best_prec_scores,
                'best_recalls': best_recall_scores,
                'best_f1_scores': best_f1_scores}
    else:
        return {'best_thresholds': best_thresholds,
                'mean_best_precision' : best_prec_scores['mean'],
                'std_best_precision' : best_prec_scores['std'],
                'mean_best_recall' : best_recall_scores['mean'],
                'std_best_recall' : best_recall_scores['std'],
                'mean_best_f1_scores' : best_f1_scores['mean'],
                'std_best_f1_scores' : best_f1_scores['std']}