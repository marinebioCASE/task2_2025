from datetime import datetime
from metrics_utils import compute_best_pr_and_f1

def to_datetime(time_, format, with_zone=False):
    time_ = time_.split('.')[0] if '.wav' in time_ else time_

    if with_zone and '%z' in format:
        return datetime.strptime(time_, format)
    else:
        if '%z' in format:
            dt = datetime.strptime(time_, format)
            return dt.replace(tzinfo=None)
        else:
            return datetime.strptime(time_, format)


def binarize_preds():
    pass

def binary_to_timestamp():
    pass
