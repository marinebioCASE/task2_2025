import os

import torch
from torch.utils.data import DataLoader
from torchvision import models

from audio_dataset import AudioDataset

def load_data(args, load_info=False, val_only=False, test_only=False): # TODO surely a better way to do so by moding AudioDataset
    print('[INFO]: Loading data')

    if val_only:
        val_set = AudioDataset(args, mode='validation', load_info=load_info)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)
        return val_loader

    #if test_only:
        #test_set = AudioDataset(args, mode='test', load_info=load_info)
        #test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)
        #return test_loader

    else:
        train_set = AudioDataset(args, mode='train', load_info=load_info)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

        val_set = AudioDataset(args, mode='validation', load_info=load_info)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=True)

        #test_set = AudioDataset(args, mode='test', load_info=load_info)
        #test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

        return train_loader, val_loader

def init_resnet18(args):
    if args.pretrained:
        print("[INFO]: Loading pre-trained weights")
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    elif not args.pretrained:
        print("[INFO]: Not loading pre-trained weights")
        model = models.resnet18(weights=None)
    if args.fine_tune:
        print("[INFO]: Fine-tuning all layers...")
        for params in model.parameters():
            params.requires_grad = True
    elif not args.fine_tune:
        print("[INFO]: Freezing hidden layers...")
        for params in model.parameters():
            params.requires_grad = False

    model.fc = torch.nn.Linear(512, args.n_classes)
    return model

def load_from_ckpt(args, model, device, optimizer=None):
    """
    Load a model from a checkpoint.
    If optimizer == None, will behave for inference and won't return additional params needed to resume training
    """
    path = os.path.join(args.outputs_path, args.modelckpt)
    print(f'[INFO]: Checkpointing from {path}')

    checkpoint = torch.load(path, weights_only=False, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer is None: # loading to infer, not to resume training -> no need to load optimizer, loss and monitored metrics
        return model

    else:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_f1 = checkpoint['best_f1']
        best_val_loss = checkpoint['best_val_loss']
        patience_counter_f1 = checkpoint['patience_counter_f1']
        patience_counter_loss = checkpoint['patience_counter_loss']

        return model, optimizer, epoch, best_f1, best_val_loss, patience_counter_f1, patience_counter_loss
