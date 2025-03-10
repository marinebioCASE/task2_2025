import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import torchaudio

import dataset


def compute_confusion_matrix(ground_truth, predictions,):
    conf_matrix = pd.DataFrame(columns=['tp', 'fp', 'fn'], index=[0, 1, 2])
    for class_id, class_predictions in predictions.groupby('annotation'):
        ground_truth_class = ground_truth.loc[ground_truth.Tags == class_id]
        conf_matrix.loc[class_id, 'tp'] = ground_truth_class['detected'].sum()
        conf_matrix.loc[class_id, 'fp'] = len(class_predictions) - class_predictions['correct'].sum()
        conf_matrix.loc[class_id, 'fn'] = len(ground_truth_class) - ground_truth_class['detected'].sum()

    conf_matrix['recall'] = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'])
    conf_matrix['precision'] = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fp'])

    return conf_matrix


def run(predictions_path, ground_truth_path, iou_threshold=0.5, conf=0.9):
    ground_truth = pd.read_csv(ground_truth_path)
    predictions = pd.read_csv(predictions_path)
    ground_truth['detected'] = 0
    predictions['correct'] = 0

    for wav_path_name, wav_predictions in predictions.groupby('filename'):
        ground_truth_wav = ground_truth.loc[ground_truth['filename'] == wav_path_name]
        for class_id, class_predictions in wav_predictions.groupby('annotation'):
            ground_truth_wav_class = ground_truth_wav.loc[ground_truth_wav['annotation'] == class_id]
            ground_truth_not_detected = ground_truth_wav_class.loc[ground_truth_wav_class.max_iou < iou_threshold]
            for i, row in tqdm(class_predictions.iterrows(), total=len(class_predictions)):
                # For each row, compute the minimum end and maximum start with all the ground truths
                min_end = np.minimum(row['end_datetime'], ground_truth_not_detected['end_datetime'])
                max_start = np.maximum(row['start_datetime'], ground_truth_not_detected['start_datetime'])
                inter = (min_end - max_start).clip(0)
                union = (row['end_datetime'] - row['start_datetime']) + (
                    (ground_truth_not_detected['end_datetime'] - ground_truth_not_detected['start_datetime'])) - inter
                iou = inter / union

                # Save the maximum iou for that prediction
                if iou.max() > iou_threshold:
                    predictions.loc[i, 'correct'] = 1
                    ground_truth_index = ground_truth_not_detected.iloc[iou.argmax()].index
                    ground_truth.loc[ground_truth_index, 'detected'] = 1

    conf_matrix = compute_confusion_matrix(ground_truth, predictions)

    for class_id, class_predictions in ground_truth.groupby('Tags'):
        #print(class_id, class_predictions)
        precision_list[class_id].append(conf_matrix.loc[class_id, 'precision'])
        recall_list[class_id].append(conf_matrix.loc[class_id, 'recall'])
        false_alarm_list[class_id].append(conf_matrix.loc[class_id, 'false_alarm'])
        tcr_list[class_id].append(conf_matrix.loc[class_id, 'tcr'])
        nmr_list[class_id].append(conf_matrix.loc[class_id, 'nmr'])
        cmr_list[class_id].append(conf_matrix.loc[class_id, 'cmr'])
        f_list[class_id].append(conf_matrix.loc[class_id, 'f'])


        total_conf_matrix = compute_confusion_matrix(iou_threshold, ground_truth, predictions, 0.5)
        total_conf_matrix.loc[3] = total_conf_matrix.mean()
        total_conf_matrix.to_csv(predictions_folder.joinpath('conf_mat_0.5.csv'))

        total_conf_matrix = pd.DataFrame(columns=total_conf_matrix.columns, index=[0, 1, 2])
        selected_confidences = {0: 0.16, 1: 0.6, 2: 0.15}
        for class_id in int2class.keys():
            conf_matrix = compute_confusion_matrix(iou_threshold, ground_truth, predictions, selected_confidences[class_id])
            total_conf_matrix.loc[class_id] = conf_matrix.loc[class_id]
        total_conf_matrix.loc[3] = total_conf_matrix.mean()
        total_conf_matrix.to_csv(predictions_folder.joinpath('conf_mat_selected.csv'))


if __name__ == '__main__':
    predictions_csv_path = input('Where are the predictions in csv format?')
    ground_truth_csv_path = input('Where are the ground truth in csv format?')
    run(predictions_csv_path, ground_truth_csv_path)
