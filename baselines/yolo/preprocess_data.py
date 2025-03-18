import json
import shutil
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import torch
import torch.nn.functional as F_general
import torchaudio
import torchaudio.functional as F
from PIL import Image
from tqdm import tqdm
import datetime

pd.set_option('future.no_silent_downcasting', True)


class YOLODataset:
    def __init__(self, config, path_to_dataset):
        # Spectrogram settings
        self.duration = config['duration']
        self.overlap = config['overlap']  # overlap of the chunks in %
        self.desired_fs = config['desired_fs']
        self.channel = config['channel']
        self.log = config['log']
        self.color = config['color']

        self.nfft = config['nfft']
        self.win_len = config['win_len']
        self.hop_len = config['hop_len']
        self.win_overlap = self.win_len - self.hop_len

        # Get the color map by name:
        #self.cmap = plt.get_cmap(config['cmap'])

        # Path to dataset
        self.path_to_dataset = pathlib.Path(path_to_dataset)
        self.wavs_folder = self.path_to_dataset.joinpath('raw')
        self.annotations_folder = self.path_to_dataset.joinpath('annotations')
        self.images_folder = self.path_to_dataset.joinpath('images')
        self.labels_folder = self.path_to_dataset.joinpath('labels')
        if not self.images_folder.exists():
            os.mkdir(self.images_folder)
            os.mkdir(self.labels_folder)
            os.mkdir(self.labels_folder.joinpath('backgrounds'))

        self.F_MIN = 0
        self.blocksize = int(self.duration * self.desired_fs)
        self.config = config

    def __setitem__(self, key, value):
        if key in self.config.keys():
            self.config[key] = value
        self.__dict__[key] = value

    def save_config(self, config_path):
        with open(config_path, 'w') as f:
            json.dump(self.config, f)

    def create_dataset(self, class_encoding):
        indices_per_deployment = self.convert_challenge_annotations_to_yolo(class_encoding=class_encoding)
        selected_samples = self.select_background_labels(indices_per_deployment)
        self.create_spectrograms(selected_samples=selected_samples)

    def select_background_labels(self, indices_per_deployment):
        for dataset, indices_dataset in indices_per_deployment.items():
            label_indices = indices_dataset['labels']
            background_indices = indices_dataset['background']
            n_labels = len(label_indices)
            print(f'There are {n_labels} labels in the dataset {dataset}')

            selected_background = background_indices.sample(n=n_labels)
            indices_per_deployment[dataset]['selected_background'] = selected_background

            for selection in selected_background:
                shutil.move(self.labels_folder.joinpath('backgrounds', selection + '.txt'),
                            self.labels_folder.joinpath(selection + '.txt'))

        return indices_per_deployment

    def create_spectrograms(self, selected_samples, overwrite=True):
        # First, create all the images
        print(self.wavs_folder)
        for dataset, indices_dataset in selected_samples:
            selected_indices = indices_dataset['selected_background'] + indices_dataset['labels']
            for sample_i in selected_indices:
                img_path = self.images_folder.joinpath(sample_i + '.png')
                wav_name = sample_i.split('_')[:2].join('_')
                wav_path = self.wavs_folder.joinpath(wav_name + '.wav')
                i = float(sample_i.split('_')[-1])
                if overwrite or (not img_path.exists()):
                    start_chunk = int(i * self.blocksize)
                    chunk, fs = torchaudio.load(wav_path, normalize=True, frame_offset=start_chunk,
                                                num_frames=self.blocksize)
                    chunk = chunk[0, :]

                    if len(chunk) < self.blocksize:
                        chunk = F_general.pad(chunk, (0, self.blocksize - len(chunk)))
                    img, f = self.create_chunk_spectrogram(chunk)

                    if self.log:
                        fig, ax = plt.subplots()
                        ax.pcolormesh(img[:, :, ::-1])
                        ax.set_yscale('symlog')
                        plt.axis('off')
                        plt.ylim(bottom=3)
                        plt.savefig(img_path, bbox_inches='tight', pad_inches=0)
                    else:
                        Image.fromarray(np.flipud(img)).save(img_path)
                    plt.close()
                i += self.overlap

    def create_chunk_spectrogram(self, chunk):
        sos = scipy.signal.iirfilter(20, [5, 124], rp=None, rs=None, btype='band',
                                     analog=False, ftype='butter', output='sos',
                                     fs=self.desired_fs)
        chunk = scipy.signal.sosfilt(sos, chunk)
        f, t, sxx = scipy.signal.spectrogram(chunk, fs=self.desired_fs, window=('hann'),
                                             nperseg=self.win_len,
                                             noverlap=self.win_overlap, nfft=self.nfft,
                                             detrend=False,
                                             return_onesided=True, scaling='density', axis=-1,
                                             mode='magnitude')
        sxx = 1 - sxx
        per = np.percentile(sxx.flatten(), 98)
        sxx = (sxx - sxx.min()) / (per - sxx.min())
        sxx[sxx > 1] = 1
        img = np.array(sxx * 255, dtype=np.uint8)
        return img, f

    def convert_challenge_annotations_to_yolo(self, class_encoding=None):
        """
        :param annotations_file:
        :param labels_to_exclude: list
        :param class_encoding: should be a dict with the name of the Tag as a key and an int as the value, for the
        yolo classes
        :return:
        """
        f_bandwidth = (self.desired_fs / 2) - self.F_MIN
        indices_per_deployment = {}
        for selections_path in self.annotations_folder.glob('*.csv'):
            background_indices = []
            labels_indices = []
            selections = pd.read_csv(selections_path, parse_dates=['start_datetime', 'end_datetime'])
            selections.loc[selections['low_frequency'] < self.F_MIN, 'low_frequency'] = self.F_MIN
            selections['height'] = (selections['high_frequency'] - selections['low_frequency']) / f_bandwidth

            # The y is from the TOP!
            selections['y'] = 1 - (selections['high_frequency'] / f_bandwidth)

            # compute the width in pixels
            selections['width'] = (
                        (selections['end_datetime'] - selections['start_datetime']).dt.total_seconds() / self.duration)

            # Deal with datetime
            selections['start_datetime_wav'] = pd.to_datetime(selections['filename'].apply(lambda y: y.split('.')[0]),
                                                              format='%Y-%m-%dT%H-%M-%S_%f')
            selections['start_datetime_wav'] = selections['start_datetime_wav'].dt.tz_localize('UTC')
            selections['start_seconds'] = (selections.start_datetime - selections.start_datetime_wav).dt.total_seconds()
            selections['end_seconds'] = (selections.end_datetime - selections.start_datetime_wav).dt.total_seconds()

            pbar = tqdm(total=len(selections['filename'].unique()))

            dataset_name = selections.iloc[0].dataset
            for wav_name, wav_selections in selections.groupby('filename'):
                wav_file_path = self.wavs_folder.joinpath(dataset_name, wav_name)
                waveform_info = torchaudio.info(wav_file_path)

                i = 0.0
                while (i * self.duration + self.duration / 2) < (waveform_info.num_frames / waveform_info.sample_rate):
                    start_seconds = i * self.duration
                    end_seconds = i * self.duration + self.duration

                    start_mask = (wav_selections['start_seconds'] >= start_seconds) & (wav_selections[
                                                                                           'start_seconds'] <= end_seconds)
                    end_mask = (wav_selections['start_seconds'] >= start_seconds) & (wav_selections[
                                                                                         'end_seconds'] <= end_seconds)
                    chunk_selection = wav_selections.loc[start_mask | end_mask]

                    chunk_selection = chunk_selection.assign(
                        x=(chunk_selection['start_seconds'] - i * self.duration) / self.duration)

                    chunk_selection.loc[
                        (chunk_selection['width'] + chunk_selection['x']) > 1, 'width'] = 1 - chunk_selection['x']

                    chunk_selection.loc[chunk_selection['x'] < 0, 'width'] = chunk_selection['width'] - chunk_selection[
                        'x']
                    chunk_selection.loc[chunk_selection['x'] < 0, 'x'] = 0

                    # Save the chunk detections so that they are with the yolo format
                    # <class > < x > < y > < width > < height >
                    chunk_selection['x'] = (chunk_selection['x'] + chunk_selection['width'] / 2)
                    chunk_selection['y'] = (chunk_selection['y'] + chunk_selection['height'] / 2)

                    chunk_selection.x = chunk_selection.x.clip(lower=0, upper=1)
                    chunk_selection.y = chunk_selection.y.clip(lower=0, upper=1)

                    chunk_selection = chunk_selection.replace(to_replace=class_encoding).infer_objects(copy=False)
                    new_name = dataset_name + wav_name.replace('.wav', '_%s' % i)

                    if len(chunk_selection) > 0:
                        labels_indices.append(new_name)
                        label_path = self.labels_folder.joinpath(new_name + '.txt')
                    else:
                        background_indices.append(new_name)
                        label_path = self.labels_folder.joinpath('backgrounds', new_name + '.txt')

                    chunk_selection[[
                        'annotation',
                        'x',
                        'y',
                        'width',
                        'height']].to_csv(label_path, header=None, index=None, sep=' ', mode='w')
                    # Add the station if the image adds it as well!
                    i += self.overlap
                pbar.update(1)
            indices_per_deployment[dataset_name] = {'background': background_indices, 'labels': labels_indices}
            pbar.close()

        return indices_per_deployment

    def all_predictions_to_dataframe(self, labels_folder, overwrite=True):
        detected_foregrounds = []
        f_bandwidth = (self.desired_fs / 2) - self.F_MIN
        if self.split_folders:
            wav_folder = self.wavs_folder.joinpath(labels_folder.name)
        else:
            wav_folder = self.wavs_folder
        for txt_label in tqdm(labels_folder.glob('*.txt'), total=len(list(labels_folder.glob('*.txt')))):
            name_parts = txt_label.name.split('_')
            wav_name = '_'.join(name_parts[:-1]) + '.wav'
            original_wav = wav_folder.joinpath(wav_name)
            offset_seconds = float(name_parts[-1].split('.txt')[0])
            detections = pd.read_table(txt_label, header=None, sep=' ', names=['class', 'x', 'y',
                                                                               'width', 'height', 'confidence'])
            detections['folder'] = str(original_wav.parent)
            detections['wav'] = str(original_wav)
            detections['wav_name'] = wav_name
            detections['start_seconds'] = (detections.x - detections.width / 2 + offset_seconds) * self.duration
            detections['image'] = txt_label.name.replace('.txt', '')

            # for _, row in detections.iterrows():
            detected_foregrounds.extend(detections.values)
            # detected_foregrounds = pd.concat([detected_foregrounds, detections], ignore_index=True)
        detected_foregrounds = np.stack(detected_foregrounds)
        detected_foregrounds = pd.DataFrame(detected_foregrounds, columns=detections.columns)
        detected_foregrounds['duration'] = detected_foregrounds.width * self.duration
        detected_foregrounds['min_freq'] = (1 - (
                    detected_foregrounds.y + detected_foregrounds.height / 2)) * f_bandwidth
        detected_foregrounds['max_freq'] = (1 - (
                    detected_foregrounds.y - detected_foregrounds.height / 2)) * f_bandwidth
        return detected_foregrounds

    def convert_yolo_detections_to_csv(self, predictions_folder, add_station_name=False, min_conf=None):
        # Convert to DataFrame
        labels_folder = predictions_folder.joinpath('labels')
        if self.split_folders:
            folders_list = [f for f in labels_folder.glob('*') if f.is_dir()]
        else:
            folders_list = [labels_folder]
        for f in folders_list:
            print('processing folder %s' % f.name)
            if self.split_folders:
                roi_path = predictions_folder.joinpath('roi_detections_clean_%s.txt' % f.name)
            else:
                roi_path = predictions_folder.joinpath('roi_detections_clean.txt')
            if not roi_path.exists():
                detected_foregrounds = self.all_predictions_to_dataframe(labels_folder=f)
                if min_conf is not None:
                    detected_foregrounds = detected_foregrounds.loc[detected_foregrounds.confidence >= min_conf]

                # Convert to RAVEN format
                columns = ['Selection', 'View', 'Channel', 'Begin File', 'End File', 'start_datetime', 'end_datetime',
                           'Beg File Samp (samples)', 'End File Samp (samples)', 'low_frequency', 'high_frequency',
                           'Tags']

                # convert the df to Raven format
                detected_foregrounds['low_frequency'] = detected_foregrounds['min_freq']
                detected_foregrounds['high_frequency'] = detected_foregrounds['max_freq']
                detected_foregrounds['Tags'] = detected_foregrounds['class']

                detected_foregrounds.loc[detected_foregrounds['low_frequency'] < 0, 'low_frequency'] = 0
                detected_foregrounds = detected_foregrounds.loc[
                    detected_foregrounds['low_frequency'] <= self.desired_fs / 2]
                detected_foregrounds.loc[detected_foregrounds['high_frequency'] > self.desired_fs / 2,
                'high_frequency'] = self.desired_fs / 2

                detected_foregrounds['View'] = 'Spectrogram 1'
                detected_foregrounds['Channel'] = 1
                detected_foregrounds['Begin File'] = detected_foregrounds['wav_name']
                detected_foregrounds['End File'] = detected_foregrounds['wav_name']
                detected_foregrounds['start_datetime'] = detected_foregrounds['start_seconds']
                detected_foregrounds['end_datetime'] = detected_foregrounds['start_seconds'] + detected_foregrounds[
                    'duration']

                detected_foregrounds['fs'] = np.nan
                detected_foregrounds['cummulative_sec'] = np.nan

                cummulative_seconds = 0
                if self.split_folders:
                    wavs_f_folder = self.wavs_folder.joinpath(f.name)
                else:
                    wavs_f_folder = self.wavs_folder
                wavs_to_check = list(wavs_f_folder.glob('*.wav'))
                if isinstance(wavs_to_check[0], pathlib.PosixPath):
                    wavs_to_check.sort()
                for wav_file_path in wavs_to_check:
                    if add_station_name:
                        wav_path_name = wav_file_path.parent.parent.parent.name.split('_')[0] + '_' + wav_file_path.name
                    else:
                        wav_path_name = wav_file_path.name

                    waveform_info = torchaudio.info(wav_file_path)
                    mask = detected_foregrounds['wav_name'] == wav_path_name
                    detected_foregrounds.loc[mask, 'cummulative_sec'] = cummulative_seconds
                    detected_foregrounds.loc[mask, 'fs'] = waveform_info.sample_rate
                    cummulative_seconds += waveform_info.num_frames / waveform_info.sample_rate

                detected_foregrounds['Beg File Samp (samples)'] = (detected_foregrounds['start_datetime']
                                                                   * detected_foregrounds['fs']).astype(int)
                detected_foregrounds['End File Samp (samples)'] = (detected_foregrounds['end_datetime']
                                                                   * detected_foregrounds['fs']).astype(int)
                detected_foregrounds['start_datetime'] = detected_foregrounds['start_datetime'] + detected_foregrounds[
                    'cummulative_sec']
                detected_foregrounds['end_datetime'] = detected_foregrounds['end_datetime'] + detected_foregrounds[
                    'cummulative_sec']

                detected_foregrounds = detected_foregrounds.sort_values('start_datetime')
                detected_foregrounds = detected_foregrounds.reset_index(names='Selection')
                detected_foregrounds['Selection'] = detected_foregrounds['Selection'] + 1

                clean_detections = pd.DataFrame()
                for _, class_detections in detected_foregrounds.groupby('Tags'):
                    clean_detections_class = self.join_overlapping_detections_in_chunks(class_detections)
                    clean_detections = pd.concat([clean_detections, clean_detections_class])
                clean_detections['Selection'] = clean_detections['Selection'].astype(int)

                clean_detections.to_csv(roi_path, sep='\t', index=False)
            else:
                clean_detections = pd.read_table(roi_path)
        return clean_detections, roi_path

    @staticmethod
    def join_overlapping_detections(raven_detections_df, iou_threshold=0.5):
        # join all the detections overlapping an iou more than the threshold %
        selected_ids = []
        for wav_file_name, wav_selections in tqdm(raven_detections_df.groupby('Begin File'),
                                                  total=len(raven_detections_df['Begin File'].unique())):
            already_joined_ids = []
            if len(wav_selections) > 1:
                # wav_selections = wav_selections.sort_values('start_datetime')
                for i, one_selection in wav_selections.iterrows():
                    selections_to_check = wav_selections.loc[~wav_selections.index.isin(already_joined_ids)].copy()
                    if i not in already_joined_ids:
                        min_end = np.minimum(one_selection['end_datetime'], selections_to_check['end_datetime'])
                        max_start = np.maximum(one_selection['start_datetime'], selections_to_check['start_datetime'])
                        max_bottom = np.maximum(one_selection['low_frequency'], selections_to_check['low_frequency'])
                        min_top = np.minimum(one_selection['high_frequency'], selections_to_check['high_frequency'])
                        # possible_overlaps = selections_to_check.loc[(min_end > max_start) & (min_top > max_bottom)]
                        inter = (min_end - max_start).clip(0) * (min_top - max_bottom).clip(0)
                        union = ((one_selection['end_datetime'] - one_selection['start_datetime']) *
                                 (one_selection['high_frequency'] - one_selection['low_frequency'])) + \
                                ((selections_to_check['end_datetime'] - selections_to_check['start_datetime']) *
                                 (selections_to_check['high_frequency'] - selections_to_check['low_frequency'])) - inter
                        iou = inter / union
                        overlapping_selections = selections_to_check.loc[iou > iou_threshold]
                        if len(overlapping_selections) > 1:
                            already_joined_ids = np.concatenate([already_joined_ids,
                                                                 overlapping_selections.index.values])

                            raven_detections_df.loc[i, 'start_datetime'] = overlapping_selections[
                                'start_datetime'].min()
                            raven_detections_df.loc[i, 'end_datetime'] = overlapping_selections['end_datetime'].max()
                            raven_detections_df.loc[i, 'Beg File Samp (samples)'] = overlapping_selections[
                                'Beg File Samp (samples)'].max()
                            raven_detections_df.loc[i, 'End File Samp (samples)'] = overlapping_selections[
                                'End File Samp (samples)'].max()
                            raven_detections_df.loc[i, 'low_frequency'] = overlapping_selections['low_frequency'].min()
                            raven_detections_df.loc[i, 'high_frequency'] = overlapping_selections[
                                'high_frequency'].max()
                            raven_detections_df.loc[i, 'confidence'] = overlapping_selections['confidence'].max()
                        else:
                            already_joined_ids = np.concatenate([already_joined_ids, [i]])
                        selected_ids.append(i)
            else:
                selected_ids += [wav_selections.index.values[0]]
        cleaned_detections = raven_detections_df.loc[selected_ids]
        return cleaned_detections

    def join_overlapping_detections_in_chunks(self, raven_detections_df, iou_threshold=0.5):
        # join all the detections overlapping an iou more than the threshold %
        selected_rows = []

        for chunk_n, chunk in tqdm(raven_detections_df.groupby('Begin File'),
                                   total=len(raven_detections_df.groupby('Begin File'))):
            already_joined_ids = []
            if len(chunk) > 1:
                # wav_selections = wav_selections.sort_values('start_datetime')
                for i, one_selection in chunk.iterrows():
                    selections_to_check = chunk.loc[~chunk.index.isin(already_joined_ids)].copy()
                    if i not in already_joined_ids:
                        min_end = np.minimum(one_selection['end_datetime'], selections_to_check['end_datetime'])
                        max_start = np.maximum(one_selection['start_datetime'], selections_to_check['start_datetime'])
                        max_bottom = np.maximum(one_selection['low_frequency'], selections_to_check['low_frequency'])
                        min_top = np.minimum(one_selection['high_frequency'], selections_to_check['high_frequency'])
                        # possible_overlaps = selections_to_check.loc[(min_end > max_start) & (min_top > max_bottom)]
                        inter = (min_end - max_start).clip(0) * (min_top - max_bottom).clip(0)
                        union = ((one_selection['end_datetime'] - one_selection['start_datetime']) *
                                 (one_selection['high_frequency'] - one_selection['low_frequency'])) + \
                                ((selections_to_check['end_datetime'] - selections_to_check['start_datetime']) *
                                 (selections_to_check['high_frequency'] - selections_to_check['low_frequency'])) - inter
                        iou = inter / union
                        overlapping_selections = selections_to_check.loc[iou > iou_threshold]
                        if len(overlapping_selections) > 1:
                            already_joined_ids = np.concatenate([already_joined_ids,
                                                                 overlapping_selections.index.values])

                            one_selection['start_datetime'] = overlapping_selections['start_datetime'].min()
                            one_selection['end_datetime'] = overlapping_selections['end_datetime'].max()
                            one_selection['Beg File Samp (samples)'] = overlapping_selections[
                                'Beg File Samp (samples)'].max()
                            one_selection['End File Samp (samples)'] = overlapping_selections[
                                'End File Samp (samples)'].max()
                            one_selection['low_frequency'] = overlapping_selections['low_frequency'].min()
                            one_selection['high_frequency'] = overlapping_selections[
                                'high_frequency'].max()
                            one_selection['confidence'] = overlapping_selections['confidence'].max()
                        else:
                            already_joined_ids = np.concatenate([already_joined_ids, [i]])
                        selected_rows.append(one_selection.values)
            else:
                selected_rows.append(chunk.iloc[0].values)

        cleaned_detections = np.stack(selected_rows)
        cleaned_detections = pd.DataFrame(cleaned_detections, columns=chunk.columns)

        return cleaned_detections

    def all_snippets(self, detected_foregrounds, labels_to_exclude=None):
        file_list = os.listdir(self.wavs_folder)
        for i, row in tqdm(detected_foregrounds.iterrows(), total=len(detected_foregrounds)):
            if 'wav' not in detected_foregrounds.columns:
                wav_path = self.wavs_folder.joinpath(row['Begin File'])
            else:
                wav_path = row['wav']
            waveform_info = torchaudio.info(wav_path)

            # If the selection is in between two files, open both and concatenate them
            if row['Beg File Samp (samples)'] > row['End File Samp (samples)']:
                waveform1, fs = torchaudio.load(wav_path,
                                                frame_offset=row['Beg File Samp (samples)'],
                                                num_frames=waveform_info.num_frames - row[
                                                    'Beg File Samp (samples)'])

                wav_path2 = self.wavs_folder.joinpath(file_list[file_list.index(row['Begin File']) + 1])
                waveform2, fs = torchaudio.load(wav_path2,
                                                frame_offset=0,
                                                num_frames=row['End File Samp (samples)'])
                waveform = torch.cat([waveform1, waveform2], -1)
            else:
                waveform, fs = torchaudio.load(wav_path,
                                               frame_offset=row['Beg File Samp (samples)'],
                                               num_frames=row['End File Samp (samples)'] - row[
                                                   'Beg File Samp (samples)'])
            if waveform_info.sample_rate > self.desired_fs:
                waveform = F.resample(waveform=waveform, orig_freq=fs, new_freq=self.desired_fs)[self.channel, :]
            else:
                waveform = waveform[self.channel, :]

            yield i, row, waveform, fs


if __name__ == '__main__':
    path_to_dataset = input('Where is the dataset folder?')

    config_path = './dataset_config.json'
    f = open(config_path)
    config = json.load(f)

    ds = YOLODataset(config, path_to_dataset)
    ds.create_dataset(class_encoding=config['class_encoding'])
