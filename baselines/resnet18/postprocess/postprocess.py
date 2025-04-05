import os
import ast
import numpy as np
import pandas as pd
import torch

from args import args
from postprocess_utils import to_datetime, binarize_preds, binary_to_timestamp

def postprocess(args):
    preds_path = os.path.join(args.outputs_path, args.preds_path)
    df = pd.read_csv(preds_path)

    preds_bin = binarize_preds(df, args, save_name='preds_val_bin.csv')

    df = binary_to_timestamp(preds_bin, args, save_name='preds_val_timestamp.csv')
    print(df)

if __name__ == '__main__':
    postprocess(args)
