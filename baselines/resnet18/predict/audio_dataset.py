import os
import pandas as pd
import torch
import torchvision
import torchaudio
from torch.utils.data import Dataset

class AudioDataset(Dataset):

    def __init__(self, args, mode, load_info):
        self.args = args
        self.mode = mode
        self.load_info = load_info
        self.labels = args.labels

        self.audio_path = None
        self.annot_path = None
        self.annotations = None
        self._init_path_and_annot()

        self.spectroT = None
        self.amp_to_dbT = None
        self.resnet_preprocess = None
        self._init_transforms()


    def _init_path_and_annot(self):
        self.audio_path = os.path.join(self.args.data_root, self.mode, 'audio')

        if self.mode == 'train':
            self.annot_path = os.path.join(self.args.data_root, self.mode, self.args.train_annot)
            if os.path.isfile(self.annot_path):
                self.annotations = pd.read_csv(self.annot_path)
            else:

                final_df = pd.DataFrame()
                for file in [f for f in os.listdir(self.annot_path) if f.endswith('.csv')]:
                    df_tmp = pd.read_csv(os.path.join(self.annot_path, file))
                    final_df = pd.concat([final_df, df_tmp], ignore_index=True)
                self.annotations = final_df

        elif self.mode == 'validation':
            self.annot_path = os.path.join(self.args.data_root, self.mode, self.args.val_annot)
            if os.path.isfile(self.annot_path):
                self.annotations = pd.read_csv(self.annot_path)
            else:
                final_df = pd.DataFrame()
                for file in [f for f in os.listdir(self.annot_path) if f.endswith('.csv')]:
                    df_tmp = pd.read_csv(os.path.join(self.annot_path, file))
                    final_df = pd.concat([final_df, df_tmp], ignore_index=True)
                self.annotations = final_df

        elif self.mode == 'test':
            self.annot_path = os.path.join(self.args.data_root, self.mode, self.args.test_annot)
            if os.path.isfile(self.annot_path):
                self.annotations = pd.read_csv(self.annot_path)
            else:
                final_df = pd.DataFrame()
                for file in [f for f in os.listdir(self.annot_path) if f.endswith('.csv')]:
                    df_tmp = pd.read_csv(os.path.join(self.annot_path, file))
                    final_df = pd.concat([final_df, df_tmp], ignore_index=True)
                self.annotations = final_df


    def _init_transforms(self):
        self.spectroT = torchaudio.transforms.Spectrogram(
            n_fft=self.args.n_fft,
            win_length=self.args.win_size,
            hop_length=int(self.args.win_size - (self.args.win_size * self.args.overlap / 100)))

        self.amp_to_dbT = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=1000)

        self.resnet_preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.annotations)

    def gen_spectro_normalized(self, audio_path, start_offset, resample_sr, duration, db_threshold=30):
        info = torchaudio.info(audio_path)

        signal, orig_sr = torchaudio.load(audio_path,
                                          format='wav',
                                          frame_offset=int(start_offset * info.sample_rate),
                                          num_frames=int(duration * info.sample_rate))

        if orig_sr != resample_sr:
            signal = torchaudio.functional.resample(signal, orig_freq=orig_sr, new_freq=resample_sr)
        signal = signal / signal.std()

        spectro = self.amp_to_dbT(self.spectroT(signal))
        spectro.clamp_(-db_threshold, db_threshold)
        return (spectro + db_threshold) / (2 * db_threshold)


    def __getitem__(self, idx):
        dataset = self.annotations.iloc[idx, self.annotations.columns.get_loc('dataset')]
        filename = self.annotations.iloc[idx, self.annotations.columns.get_loc('filename')]
        audio_path = os.path.join(self.audio_path, dataset, filename)

        start_offset = self.annotations.iloc[idx, self.annotations.columns.get_loc('start_offset')]

        spectro = self.gen_spectro_normalized(audio_path,
                                              start_offset,
                                              resample_sr=self.args.sample_rate,
                                              duration=self.args.duration)
        spectro = spectro.expand(3, -1, -1)
        spectro = self.resnet_preprocess(spectro)

        idx_lab = [self.annotations.columns.get_loc(lab) for lab in self.labels]
        labels = self.annotations.iloc[idx, idx_lab].to_numpy().astype(float)
        labels = torch.Tensor(labels)

        if self.load_info:
            return (dataset, filename, start_offset), spectro, labels
        else:
            return spectro, labels



