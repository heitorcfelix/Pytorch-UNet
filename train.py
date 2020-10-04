import argparse
import logging
import os
import sys
import options

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, MVTecDataset
from torch.utils.data import DataLoader, random_split


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5):

    dataset_train = MVTecDataset(options.class_name, train=True)
    dataset_val = MVTecDataset(options.class_name, train=False)
    n_val = len(dataset_val)
    n_train = len(dataset_train)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    writer = SummaryWriter(comment=f'_UNet_Class_{options.class_name}_Epochs_{options.epochs}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=1e-8)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=options.lr_patience)
    criterion = nn.MSELoss()

    best_score = np.inf
    for epoch in range(epochs):
        net.train()
        actual_score = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']
                assert imgs.shape[1] == net.n_channels, \
                    f'Network has been defined with {net.n_channels} input channels, ' \
                    f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)

                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
        val_score = eval_net(net, val_loader, device)
        actual_score += val_score
        scheduler.step(val_score)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

        logging.info('Validation: {}'.format(val_score))
        writer.add_scalar('Loss/test', val_score, global_step)
        writer.add_images('masks/true', true_masks, global_step)
        writer.add_images('masks/pred', masks_pred, global_step)

        if save_cp:
            try:
                os.mkdir('checkpoints/')
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       'checkpoints/CP_last.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
            if actual_score <= best_score:
                best_score = actual_score
                torch.save(net.state_dict(),
                       'checkpoints/CP_best.pth')
                logging.info('Best checkpoint reached !')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=3, bilinear=True)
    logging.info(f'UNet Network:\n'
                f'\t{net.n_channels} input channels\n'
                f'\t{net.n_classes} output channels (classes)\n'
                f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if options.pretrained_path:
        net.load_state_dict(
            torch.load(options.pretrained_path, map_location=device)
        )
        logging.info(f'UNet Model loaded from {options.pretrained_path}')


    net.to(device=device)

    try:
        train_net(net=net,
                  epochs=options.epochs,
                  batch_size=options.batch_size,
                  lr=args.lr,
                  device=device,
                  img_scale=1.0,
                  val_percent=0)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
