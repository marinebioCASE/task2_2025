import os
import pandas as pd
from datetime import datetime
import torch

def save_results_to_df(args, all_info, all_y, all_y_hat):
    datasets, filenames, start_offsets = [], [], []

    for info in all_info:
        datasets.extend(info[0])
        filenames.extend(info[1])
        start_offsets.append(info[2].tolist())

    start_offsets = [el for sub in start_offsets for el in sub]

    all_y = torch.cat(all_y).tolist()
    all_y_hat = torch.cat(all_y_hat).tolist()

    df = pd.DataFrame({'dataset' : datasets,
                       'filename' : filenames,
                       'start_offset' : start_offsets,
                       'y' : all_y,
                       'y_hat' : all_y_hat})


    savepath = os.path.join(args.outputs_path, os.path.dirname(args.modelckpt), 'preds.csv') # custom name here
    df.to_csv(savepath, index=False)

    print(f'[INFO]: Results successfully saved to {savepath}')