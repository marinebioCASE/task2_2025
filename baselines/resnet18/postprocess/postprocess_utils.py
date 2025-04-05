import os
import ast
import pandas as pd
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed

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

def gen_timestamp(df):
    for i in range(df.shape[0]):
        start_dt = to_datetime(df['filename'][i][:-4], format='%Y-%m-%dT%H-%M-%S_%f')
        df.at[i, 'start_annot_dt'] = (start_dt + pd.to_timedelta(df.iloc[i]['start_offset'], unit='s'))
        df.at[i, 'end_annot_dt'] = (start_dt + pd.to_timedelta(df.iloc[i]['end_offset'], unit='s'))

    return df

def overlap_seeker(df_label_pos, label):
    list_merged = []

    n = df_label_pos.shape[0]

    if n == 0:
        return list_merged

    i = 0
    while i < n:
        f = df_label_pos.iloc[i]
        tmp = {'dataset': f['dataset'],
            'filename': f['filename'],
            'annotation': label,
            'start_datetime': f['start_annot_dt'],
            'end_datetime': f['end_annot_dt']}

        j = i + 1
        curr_end = f['end_annot_dt']

        while j < n and curr_end > df_label_pos.iloc[j]['start_annot_dt']:
            curr_end = max(curr_end, df_label_pos.iloc[j]['end_annot_dt'])
            j += 1

        tmp['end_datetime'] = curr_end
        list_merged.append(tmp)
        i = j
    return list_merged

def process_label(df, label):
    cols = ['dataset', 'filename', 'start_annot_dt', 'end_annot_dt', label]
    df_label_pos = df[df[label] == 1.0][cols]
    return overlap_seeker(df_label_pos, label)

def process_site_year(site_year, df, labels):
    all_merged = []
    with ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_label, df, label): label for label in labels}
        for future in as_completed(futures):
            merged = future.result()
            all_merged.extend(merged)
    return all_merged


def binarize_preds(df, args, save_name=None):
    all_y = np.array([ast.literal_eval(y) for y in df['y'].to_numpy()], dtype=float)
    all_y_hat = np.array([ast.literal_eval(y) for y in df['y_hat'].to_numpy()], dtype=float)
    all_y = torch.from_numpy(all_y).float()
    all_y_hat = torch.from_numpy(all_y_hat).float()

    prs_f1s = compute_best_pr_and_f1(all_y, all_y_hat, labels=args.labels, return_all=True)
    best_thresholds = np.array([t for t in prs_f1s['best_thresholds'].values()])

    df_threshed = pd.DataFrame(all_y_hat, columns=args.labels)
    for i, label in enumerate(args.labels):
        df_threshed[label] = (df_threshed[label] >= best_thresholds[i]).astype(float)

    final_df = pd.concat([df, df_threshed], axis=1)
    final_df['end_offset'] = final_df['start_offset'] + 5.
    final_df.drop(columns=['y', 'y_hat'], inplace=True)
    final_df = final_df[['dataset', 'filename', 'start_offset', 'end_offset'] + args.labels]
    final_df.sort_values(by=['dataset', 'filename', 'start_offset'], inplace=True) # sorting needed for binary_to_timestamp()

    print('[INFO]: Predictions binarized')
    if save_name is not None:
        save_path = os.path.join(args.outputs_path, os.path.dirname(args.preds_path), save_name)
        final_df.to_csv(save_path, index=False)
        print(f'[INFO]: Binary predicitions saved to {save_path}')
    return final_df

def binary_to_timestamp(df, args, save_name=None):
    df_w_timestamp = gen_timestamp(df)

    dataset_dict = {f: df for f, df in df_w_timestamp.groupby("dataset")}
    dataset_dict.keys()

    final_merged = []

    for site_year, df in dataset_dict.items():
        all_merged = process_site_year(site_year, df, args.labels)
        final_merged.extend(all_merged)

    final_df = pd.DataFrame(final_merged)

    print('[INFO]: Binary predictions turned to timestamps')

    save_path = os.path.join(args.outputs_path, os.path.dirname(args.preds_path), 'preds_timestamp.csv')
    final_df.to_csv(save_path, index=False)
    print(f'[INFO]: Timestamps predicitions saved to {save_path}')

    return final_df