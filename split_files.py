import pathlib
import pandas as pd
import shutil
from tqdm import tqdm
import os

VALID_DEPLOYMENTS = ['kerguelen2014', 'kerguelen2015', 'Casey2017']


def split_train_valid(output):
    labels_folder = output.joinpath('labels')
    images_folder = output.joinpath('images')
    all_files_list = list(labels_folder.glob('*.txt'))
    background_indices = []
    labels_indices = []
    print('Reading which files are background...')
    all_files_series_full = pd.DataFrame({'path': all_files_list})
    for i, file_row in tqdm(all_files_series_full.iterrows()):
        file_size = os.stat(file_row.path).st_size
        if file_size > 0:
            labels_indices.append(i)
        else:
            background_indices.append(i)

    n_labels = len(labels_indices)
    print(f'Found {n_labels} labels')
    all_files_series_full['deployment'] = all_files_series_full['path'].astype(str).apply(lambda y: y.split('/')[-1].split('_')[0])
    all_files_series_labels = all_files_series_full.loc[labels_indices]
    n_pos_dep = all_files_series_labels.value_counts('deployment')
    print(n_pos_dep)
    all_files_series = all_files_series_labels
    for dep_name, dep_files in all_files_series_full.loc[background_indices].groupby('deployment'):
        group_files = dep_files.sample(n=n_pos_dep[dep_name])
        all_files_series = pd.concat([all_files_series, group_files], ignore_index=True)

    valid_files = all_files_series.loc[all_files_series.deployment.isin(VALID_DEPLOYMENTS)]

    print('moving valid...')
    valid_folder = output.joinpath('valid')
    for _, row in tqdm(valid_files.iterrows(), total=len(valid_files)):
        valid_file = row.path
        try:
            shutil.move(valid_file, valid_folder.joinpath('labels', valid_file.name))
            img_file = images_folder.joinpath(valid_file.name.replace('.txt', '.png'))
            shutil.move(img_file, valid_folder.joinpath('images', img_file.name))
        except Exception as e:
            print(e)

    print('moving train...')
    train_folder = output.joinpath('train')
    for _, row in tqdm(all_files_series[~all_files_series.index.isin(valid_files.index)].iterrows(),
                           total=len(all_files_series) - len(valid_files)):
        train_file = row.path
        try:
            shutil.move(train_file, train_folder.joinpath('labels', train_file.name))
            img_file = images_folder.joinpath(train_file.name.replace('.txt', '.png'))
            shutil.move(img_file, train_folder.joinpath('images', img_file.name))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    output_folder = pathlib.Path(input('Where is the folder to split?:'))
    split_train_valid(output_folder)
