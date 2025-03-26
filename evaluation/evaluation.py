import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib

joining_dict = {'bma': 'bmabz',
                'bmb': 'bmabz',
                'bmz': 'bmabz',
                'bmd': 'd',
                'bpd': 'd',
                'bp20': 'bp',
                'bp20plus': 'bp'}


def compute_confusion_matrix(ground_truth, predictions, all_classes):
    conf_matrix = pd.DataFrame(columns=['tp', 'fp', 'fn'], index=ground_truth.annotation.unique())
    for class_id in all_classes:
        ground_truth_class = ground_truth.loc[ground_truth.annotation == class_id]
        class_predictions = predictions.loc[predictions.annotation == class_id]
        conf_matrix.loc[class_id, 'tp'] = ground_truth_class['detected'].sum()
        conf_matrix.loc[class_id, 'fp'] = len(class_predictions) - class_predictions['correct'].sum()
        conf_matrix.loc[class_id, 'fn'] = len(ground_truth_class) - ground_truth_class['detected'].sum()

    conf_matrix['recall'] = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fn'])
    conf_matrix['precision'] = conf_matrix['tp'] / (conf_matrix['tp'] + conf_matrix['fp'])

    return conf_matrix


def compute_confusion_matrix_per_dataset(ground_truth, predictions, all_classes):
    for dataset_name, ground_truth_dataset in ground_truth.groupby('dataset'):
        predictions_dataset = predictions.loc[predictions.dataset == dataset_name]
        conf_matrix_dataset = compute_confusion_matrix(ground_truth_dataset, predictions_dataset, all_classes)
        yield dataset_name, conf_matrix_dataset


def join_annotations_if_dir(path_to_annotations):
    if path_to_annotations.is_dir():
        annotations_list = []
        for annotations_path in path_to_annotations.glob('*.csv'):
            annotations = pd.read_csv(annotations_path, parse_dates=['start_datetime', 'end_datetime'])
            annotations_list.append(annotations)
        total_annotations = pd.concat(annotations_list, ignore_index=True)
    else:
        total_annotations = pd.read_csv(path_to_annotations, parse_dates=['start_datetime', 'end_datetime'])

    return total_annotations


def run(predictions_path, ground_truth_path, iou_threshold=0.3):
    ground_truth = join_annotations_if_dir(ground_truth_path)
    predictions = join_annotations_if_dir(predictions_path)

    ground_truth = ground_truth.replace(joining_dict)
    predictions = predictions.replace(joining_dict)
    ground_truth['detected'] = 0
    predictions['correct'] = 0

    for (dataset_name, wav_path_name), wav_predictions in tqdm(predictions.groupby(['dataset', 'filename']),
                                               total=len(predictions.filename.unique())):
        ground_truth_wav = ground_truth.loc[ground_truth['filename'] == wav_path_name]
        for class_id, class_predictions in wav_predictions.groupby('annotation'):
            ground_truth_wav_class = ground_truth_wav.loc[ground_truth_wav['annotation'] == class_id]
            ground_truth_not_detected = ground_truth_wav_class.loc[ground_truth_wav_class.detected == 0]
            if ground_truth_not_detected.empty:
                continue
            for i, row in class_predictions.iterrows():
                # For each row, compute the minimum end and maximum start with all the ground truths
                min_end = np.minimum(row['end_datetime'], ground_truth_not_detected['end_datetime'])
                max_start = np.maximum(row['start_datetime'], ground_truth_not_detected['start_datetime'])
                inter = (min_end - max_start).dt.total_seconds().clip(0)
                union = (row['end_datetime'] - row['start_datetime']).total_seconds() + (
                    (ground_truth_not_detected['end_datetime'] - ground_truth_not_detected['start_datetime']).dt.total_seconds()) - inter
                iou = inter / union

                # Save the maximum iou for that prediction
                if iou.max() > iou_threshold:
                    predictions.loc[i, 'correct'] = 1
                    ground_truth_index = ground_truth_not_detected.iloc[iou.argmax()].name
                    ground_truth.loc[ground_truth_index, 'detected'] = 1

    all_classes = ground_truth.annotation.unique()
    for dataset_name, conf_matrix_dataset in compute_confusion_matrix_per_dataset(ground_truth, predictions, all_classes):
        print(f'Results dataset {dataset_name}')
        print(conf_matrix_dataset)

    conf_matrix = compute_confusion_matrix(ground_truth, predictions, all_classes)
    print('Final results')
    print(conf_matrix)


if __name__ == '__main__':
    predictions_csv_path = pathlib.Path(input('Where are the predictions in csv format?'))
    ground_truth_csv_path = pathlib.Path(input('Where are the ground truth in csv format?'))
    run(predictions_csv_path, ground_truth_csv_path)
