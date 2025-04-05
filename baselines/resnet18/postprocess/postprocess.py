import os
import pandas as pd

from args import args
from postprocess_utils import to_datetime, binarize_preds, binary_to_timestamp

def postprocess(args):
    preds_path = os.path.join(args.outputs_path, args.preds_path)
    df = pd.read_csv(preds_path)

    preds_bin = binarize_preds(df, args, save_name='preds_val_bin.csv') # custome save_name here

    binary_to_timestamp(preds_bin, args)

if __name__ == '__main__':
    postprocess(args)
