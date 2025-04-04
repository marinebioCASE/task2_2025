import os
import torch
from datetime import datetime

from args import args
from loaders import load_data, init_resnet18
from train_utils import train_one_epoch, validate_one_epoch_f1


train_loader, val_loader = load_data(args)
next(iter(train_loader))
next(iter(val_loader))
print('[INFO]: Data loaded successfully')

model = init_resnet18(args)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
loss_fn = torch.nn.BCEWithLogitsLoss()
print('[INFO]: Model loaded successfully')

now = datetime.now().strftime('%m-%dT%H-%M')

if args.xp_name:
    #xp_name = f'{now}_{args.xp_name} <- Commented to create toy paths, but we advise to use the "now" parameters not to overwrite potentiel previous shots
    xp_name = args.xp_name
else:
    xp_name = now
print(f'[INFO]: xp name is {xp_name}')
out_path = os.path.join(args.outputs_path, xp_name)
os.makedirs(out_path, exist_ok=True)

model, train_loss = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
val_loss, best_metrics = validate_one_epoch_f1(model, val_loader, loss_fn, device, args)
print(f'[INFO]: Toy epoch runned successfully')
print(f'    train_loss: {train_loss:.4f}')
print(f'    val_loss: {val_loss:.4f}')
print(f'    f1: {best_metrics["mean_best_f1"]:.4f}')

torch.save({'epoch': 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),}, os.path.join(out_path, 'toy_model.pth'))