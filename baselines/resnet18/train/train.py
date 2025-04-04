import os
from datetime import datetime
import torch
#import wandb

from args import args
from loaders import load_data, init_resnet18, load_from_ckpt
from train_utils import train_one_epoch, validate_one_epoch_f1

def main():
    # wandb.login()

    train_loader, val_loader = load_data(args)
    model = init_resnet18(args)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO]: device is {device}')

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    ckpt_epoch = 0
    best_f1 = 0.
    best_val_loss = float('inf')
    patience_counter_f1 = 0
    patience_counter_loss = 0

    if args.modelckpt:
        model, optimizer, ckpt_epoch, best_f1, best_val_loss, patience_counter_f1, patience_counter_loss = load_from_ckpt(args, model, device, optimizer)
        print(f'[INFO]: Training will resume @ epoch {ckpt_epoch} with patience counter f1 {patience_counter_f1} / {args.patience} and val loss {patience_counter_loss}/{args.patience}')


    now = datetime.now().strftime('%m-%dT%H-%M')
    if args.xp_name:
        xp_name = f'{now}_{args.xp_name}'
    else:
        xp_name = now
    print(f'[INFO]: exp name is {xp_name}')
    out_path = os.path.join(args.outputs_path, xp_name)
    os.makedirs(out_path, exist_ok=True)
    # wandb.init(project='biodcase', name=out_path)

    for epoch in range(ckpt_epoch, args.n_epochs):
        print(f'[INFO]: EPOCH {epoch}')

        model, train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, best_metrics = validate_one_epoch_f1(model, val_loader, loss_fn, device, args)
        to_log = {'val_loss': val_loss,
                   'mean_val_precision': best_metrics['mean_best_prec'],
                   'mean_val_recall': best_metrics['mean_best_recall'],
                   'mean_val_f1': best_metrics['mean_best_f1']}
        #wandb.log(to_log)

        if best_metrics['mean_best_f1'] > best_f1:
            best_f1 = best_metrics['mean_best_f1']
            patience_counter_f1 = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'best_val_loss': best_val_loss,
                'patience_counter_f1': patience_counter_f1,
                'patience_counter_loss': patience_counter_loss,}, os.path.join(out_path, 'best_val_f1_model.pth'))
            print(f'[INFO]: New best model with val f1 {best_f1:.3f} saved @ epoch {epoch}')

        else:
            patience_counter_f1 += 1
            print(f'[INFO]: No improvement in val f1. Early stopping counter f1: {patience_counter_f1}/{args.patience}')

        if val_loss < best_val_loss:
            best_val_loss= val_loss
            patience_counter_loss = 0
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
                'best_val_loss': best_val_loss,
                'patience_counter_f1': patience_counter_f1,
                'patience_counter_loss': patience_counter_loss}, os.path.join(out_path, 'best_val_loss_model.pth'))
            print(f'[INFO]: New best model with val loss {best_val_loss:.3f} saved @ epoch {epoch}')
        else:
            patience_counter_loss += 1
            print(
                f'[INFO]: No improvement in val loss. Early stopping counter loss: {patience_counter_loss}/{args.patience}')

        if (patience_counter_f1 >= args.patience) and (patience_counter_loss >= args.patience): # early stopping
            print('[INFO]: Early stopping triggered. Stopping training.')
            break

if __name__ == '__main__':
    main()
