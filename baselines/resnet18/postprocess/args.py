import argparse

parser = argparse.ArgumentParser()

# === PATHING & NAMING ===
parser.add_argument('--data_root', default='../../../../biodcase_development_set/', type=str, help='Audio & annotations root directory')
# Root is defined as if a same root dir was containing data and the clone of the Git repo, feel free to customize it

parser.add_argument('--outputs_path', default='../outputs', type=str)
parser.add_argument('--preds_path', type=str, required=True) # assuming they're in the outputs/ dir

parser.add_argument('--n_classes', type=int, default=7)
# WARNING : labels must be in the CSV annot files passed above, no conversion here
# 3 -> ['abz', 'd, 'bp']
# 7 -> ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus']
# 10 -> ['bma','bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus', 'abz', 'd, 'bp']

args = parser.parse_args()

if args.n_classes == 3: #TODO gérer ça
    args.labels = ['abz', 'd', 'bp']
elif args.n_classes == 7:
    args.labels = ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus']
elif args.n_classes == 10:
    args.labels = ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus', 'abz', 'd', 'bp']