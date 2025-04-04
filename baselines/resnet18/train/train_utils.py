#import wandb
from tqdm import tqdm
import torch
import torch.nn as nn

from metrics_utils import compute_best_pr_and_f1

def train_one_epoch(model, loader, loss_fn, optimizer, device):
    print('[INFO]: Training...')
    train_loss = 0.

    model.train()

    for i, (X, y) in tqdm(enumerate(loader)):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()

        y_hat = model(X)
        loss = loss_fn(y_hat, y)
        train_loss += loss.item()

        loss.backward()
        optimizer.step()

        #if ((i + 1) % 25) == 0:
            #wandb.log({"train_loss": loss})

    train_loss = train_loss / len(loader)
    return model, train_loss

def validate_one_epoch_f1(model, loader, loss_fn, device, args):
    print('[INFO]: Validating...')
    val_loss = 0.
    model.eval()

    all_y = []
    all_y_hat = []

    with torch.no_grad():
        for i, (X, y) in tqdm(enumerate(loader)):
            X, y = X.to(device), y.to(device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            val_loss += loss.item()

            y_hat_sigm = nn.Sigmoid()(y_hat)
            all_y.append(y)
            all_y_hat.append(y_hat_sigm)

    all_y = torch.cat(all_y)
    all_y_hat = torch.cat(all_y_hat)

    best_metrics = compute_best_pr_and_f1(all_y, all_y_hat, labels=args.labels)

    val_loss = val_loss / len(loader)
    return val_loss, best_metrics