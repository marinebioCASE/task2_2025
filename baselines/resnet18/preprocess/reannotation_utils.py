import os
import pandas as pd
from datetime import timedelta

import torchaudio

from split_utils import gen_offsets, overlap_ratio_dt
from time_utils import to_datetime


def process_file(file, args):
    """
    Split a file into chunks of wanted duration with desired hop between resplitting
    Reannot the splits created with the ratio of the original annotation contained in the chunk

    @param args: dict-like object - contains wanted arguments
    @param file: str - the name of the file to split and reannot
    @return: pd.DataFrame - all the splits and how much of each annot is contained in
    """
    file_path = os.path.join(args.audio_path, args.site_year, file)
    file_info = torchaudio.info(file_path)

    offsets = gen_offsets(duration=file_info.num_frames / file_info.sample_rate,
                          chunk_duration=args.chunk_len,
                          hop_duration=args.hop_len)

    df_file = pd.DataFrame(offsets, columns=['start_offset', 'end_offset'])
    df_file['filename'] = file
    df_file['dataset'] = args.site_year

    for lab in args.labels:
        df_file[lab] = 0.0

    if file not in args.annotations_dict.keys(): # not annotated -> all chunks are empty
        return df_file

    file_start = to_datetime(file, format='%Y-%m-%dT%H-%M-%S_%f')
    file_annotations = args.annotations_dict.get(file, pd.DataFrame())

    annots_list = file_annotations['annotation'].to_list()
    start_list = file_annotations['start_datetime'].to_list()
    end_list = file_annotations['end_datetime'].to_list()

    for i, (start_offset, end_offset) in enumerate(offsets):
        start_chunk_dt = file_start + timedelta(seconds=start_offset)
        end_chunk_dt = file_start + timedelta(seconds=end_offset)

        for annot, start_annot, end_annot in zip(annots_list, start_list, end_list):
            if annot in args.labels:
                overlap_ratio = overlap_ratio_dt(start_chunk_dt, end_chunk_dt, start_annot, end_annot)
                if df_file.at[i, annot] == 0:
                    df_file.at[i, annot] = overlap_ratio

    return df_file

def gen_annotations_dict(annot_file):
    """
    @param annot_file: CSV file containg the annotations
    @return: Dict[str, DataFrame] where str = filename and DataFrame = associated annotations
    """
    annotations_source = pd.read_csv(annot_file)
    annotations_dict = {f: df for f, df in annotations_source.groupby("filename")}

    for df in annotations_dict.values():
        try:
            df['start_datetime'] = pd.to_datetime(df['start_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z').dt.tz_localize(None)
            df['end_datetime'] = pd.to_datetime(df['end_datetime'], format='%Y-%m-%dT%H:%M:%S.%f%z').dt.tz_localize(None)
        except:
            pass

    return annotations_dict

def threshold_filter(args, df, strict=False):
    """
    Binarize the ratio of annotation contained in each chunk according to a threshold
    We advise thteshold = 0.5, meaning that if half of the annotation is contained in a chunk, it will be considered positive for the label

    @param args: dict-like object - contains wanted arguments
    @param df: pd.DataFrame - contains annotations
    @param strict: bool - if True, ratio must be strictly higher to threshold
    @return: pd.DataFrame - contains binary annotations
    """
    df_copy = df.copy()
    thresh = args.threshold_annot_is_pos
    labels = args.labels

    if thresh == 0 or strict:
        df_copy[labels] = (df_copy[labels] > thresh).astype(int)
    else:
        df_copy[labels] = (df_copy[labels] >= thresh).astype(int)

    return df_copy

def relabel(args, df):
    """
    Gathers or expands labels from the original 7 accoring to args.n_classes
    @param args:
    @param df: pd.DataFrame - contains binary annotations
    @return: pd.DataFrame - contains relabeled annotations
    """
    df_copy = df.copy()

    df_copy['abz'] = (df_copy['bma'] | df_copy['bmb'] | df_copy['bmz']).astype(int)
    df_copy['d'] = (df_copy['bmd'] | df_copy['bpd']).astype(int)
    df_copy['bp'] = (df_copy['bp20'] | df_copy['bp20plus']).astype(int)

    if args.n_classes == 3:
        df_copy.drop(args.labels, axis=1, inplace=True)

    return df_copy