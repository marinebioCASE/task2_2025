import os
import pandas as pd
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from reannotation_utils import gen_annotations_dict, process_file, threshold_filter, relabel
from args import args

args.audio_path = os.path.join(args.data_root, args.mode, 'audio')
args.annot_path = os.path.join(args.data_root, args.mode, 'annotations')

cols = ['dataset', 'filename', 'start_offset', 'end_offset']
args.labels = ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus']


for site_year in os.listdir(args.audio_path):
    print(f'[INFO]: --- Processing {site_year} ---')

    args.site_year = site_year
    site_year_audio_path = os.path.join(args.audio_path, site_year)
    site_year_annotations = os.path.join(args.annot_path, f'{site_year}.csv')

    args.annotations_dict = gen_annotations_dict(site_year_annotations)

    print('     Splitting and reannotating...')
    partial_process_file = partial(process_file, args=args)
    files = [f for f in os.listdir(site_year_audio_path) if f.endswith('.wav')]

    with ProcessPoolExecutor() as executor:
        dfs = list(executor.map(partial_process_file, files))
    final_df = pd.concat(dfs, ignore_index=True) # TODO reorganize columns

    if args.save_df_ratio:
        savename = f'{site_year}_chunk{args.chunk_len}_hop{args.hop_len}.csv' # custom name here
        final_df.to_csv(os.path.join(args.data_root, args.mode, args.new_annot_path, savename), index=False)
        print('     CSV with ratios saved')

    print('     Thresholding...')
    final_df_filtered = threshold_filter(args, final_df)

    if args.n_classes == 7:
        savename = f'{site_year}_chunk{int(args.chunk_len)}_hop{int(args.hop_len)}_bin.csv' # custom name here
        final_df_filtered.to_csv(os.path.join(args.data_root, args.mode, args.new_annot_path, savename), index=False)
        print('     Binary CSV saved without relabelling')

    else:
        print('     Relabelling')
        final_df_filtered_relabelled = relabel(args, final_df_filtered)
        savename = f'{site_year}_chunk{int(args.chunk_len)}_hop{int(args.hop_len)}_bin_labels{args.n_classes}.csv' # custom name here
        final_df_filtered_relabelled.to_csv(os.path.join(args.data_root, args.mode, args.new_annot_path, savename), index=False)
        print('     Binary and relabelled CSV saved')
