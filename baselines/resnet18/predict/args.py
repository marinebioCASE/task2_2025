import argparse

parser = argparse.ArgumentParser()

# === PATHING & NAMING ===
parser.add_argument('--data_root', default='../../../../biodcase_development_set/', type=str, help='Audio & annotations root directory')
# Root is defined as if a same root dir was containing data and the clone of the Git repo, feel free to customize it

parser.add_argument('--val_annot', type=str, required=True, help='Directory or file containing valid annot (if dir all files in will be concatenated)')
parser.add_argument('--test_annot', type=str, help='Directory or file containing test annot (if dir all files in will be concatenated)')

parser.add_argument('--outputs_path', default='../outputs', type=str) # Results saving
parser.add_argument('--xp_name', default='', type=str)
parser.add_argument('--modelckpt', type=str, required=True, help='args.outputs_path/<PATH + NAME.pt|.pth> containing the checkpoint')

# === AUDIO & SPECTRO ===
parser.add_argument('--sample_rate', default=250, type=int)
parser.add_argument('--duration', default=5, type=int)
parser.add_argument('--n_fft', default=512, type=int)
parser.add_argument('--win_size', default=250, type=int)
parser.add_argument('--overlap', default=98, type=int)

# === MODEL & TRAINING ===
parser.add_argument('--n_classes', default=7, type=int)
# WARNING : labels must be in the CSV annot files passed above, no conversion here
# 3 -> ['abz', 'd, 'bp']
# 7 -> ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus']
# 10 -> ['bma','bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus', 'abz', 'd, 'bp']

parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--pretrained', default=True, type=bool)
parser.add_argument('--fine_tune', default=True, type=bool)

args = parser.parse_args()

if args.n_classes == 3: #TODO gérer ça
    args.labels = ['abz', 'd', 'bp']
elif args.n_classes ==7:
    args.labels = ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus']
elif args.n_classes == 10:
    args.labels = ['bma', 'bmb', 'bmz', 'bmd', 'bpd', 'bp20', 'bp20plus', 'abz', 'd', 'bp']