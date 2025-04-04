import os
from tqdm import tqdm
import torch
import torch.nn as nn

from args import args
from loaders import load_data, init_resnet18, load_from_ckpt
from infer_utils import save_results_to_df

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO]: device is {device}')

    full_load = load_data(args, load_info=True, val_only=True)
    model = init_resnet18(args)
    model = model.to(device)
    model.eval()

    if args.modelckpt:
        model = load_from_ckpt(args, model, device)

    all_info = []
    all_y = []
    all_y_hat = []

    with torch.no_grad():
        for info, X, y in tqdm(full_load):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            y_hat_sigm = nn.Sigmoid()(y_hat)

            all_info.append(info)
            all_y.append(y)
            all_y_hat.append(y_hat_sigm)

    save_results_to_df(args, all_info, all_y, all_y_hat)

if __name__ == '__main__':
    main()