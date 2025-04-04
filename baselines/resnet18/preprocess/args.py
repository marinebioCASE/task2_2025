import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--data_root', default='../../../../biodcase_development_set/', type=str, help='Audio & annotations root directory')
# Root is defined as if a same root dir was containing data and the clone of the Git repo, feel free to customize it
parser.add_argument('--annot_bin_dir', default='', help='Dir where binarized annotations will be stored')

parser.add_argument('--mode', required=True, type=str, choices={'train', 'validation'})

parser.add_argument('--n_classes', default=7, type=int, choices={3, 7, 10})
# 3 -> ['abz', 'd, 'bp']
# 7 -> ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus']
# 10 -> ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus', 'abz', 'd', 'bp']

parser.add_argument('--chunk_len', default=5., type=float) # in seconds
parser.add_argument('--hop_len', default=2., type=float) # in seconds
parser.add_argument('--threshold_annot_is_pos', default=0.5, type=float)

parser.add_argument('--save_df_ratio', action='store_true')


args = parser.parse_args()