import argparse
import logging
import os
import sys
import options

import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from tqdm import tqdm

from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset, MVTecDataset
from torch.utils.data import DataLoader, random_split

from matplotlib import pyplot as plt
from torchvision import transforms

import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import roc_curve, auc, f1_score, average_precision_score


def otsu_threshold(histogram, scores):
    # Get the image histogram
    hist, bin_edges = np.histogram(scores, bins=25)
    # Get normalized histogram if it is required
    hist = np.divide(hist.ravel(), hist.max())
    
    # Calculate centers of bins
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2.
    
    # Iterate over all thresholds (indices) and get the probabilities w1(t), w2(t)
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    
    # Get the class means mu0(t)
    mean1 = np.cumsum(hist * bin_mids) / weight1

    # Get the class means mu1(t)
    mean2 = (np.cumsum((hist * bin_mids)[::-1]) / weight2[::-1])[::-1]
    
    inter_class_variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    
    # Maximize the inter_class_variance function val
    index_of_max_val = np.argmax(inter_class_variance)
    
    threshold = bin_mids[:-1][index_of_max_val]
    return threshold

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    unet = UNet(n_channels=3, n_classes=3, bilinear=True)
    unet.load_state_dict(torch.load('./checkpoints/CP_best.pth', map_location=device))
    unet.to(device=device)

    dataset_val_zero = MVTecDataset(options.dataset_dir + 'test/0.normal/', 1)
    dataset_val_one = MVTecDataset(options.dataset_dir + 'test/1.abnormal/', 1)

    val_loader_zero = DataLoader(dataset_val_zero, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    val_loader_one = DataLoader(dataset_val_one, batch_size=1, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    unet.eval()

    n_val_zero = len(val_loader_zero)
    n_val_one = len(val_loader_one)

    scores = []

    with tqdm(total=n_val_zero, desc='Validation Normal', unit='batch', leave=False) as pbar:
        for batch in val_loader_zero:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net_G(imgs)
                scores.append(F.mse_loss(mask_pred, true_masks).item())
            pbar.update()

    with tqdm(total=n_val_one, desc='Validation Abnormal', unit='batch', leave=False) as pbar:
        for batch in val_loader_one:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = net_G(imgs)
                scores.append(F.mse_loss(mask_pred, true_masks).item())
            pbar.update()

    scores = np.array(scores)
    labels = np.zeros(n_val_zero)
    labels = np.concatenate([labels, np.ones(n_val_one)])

    scores_dict = {}
    scores_dict['scores'] = scores
    scores_dict['labels'] = labels

    hist = pd.DataFrame.from_dict(scores_dict)

    # Filter normal and abnormal scores.
    abn_scr = hist.loc[hist.labels == 1]['scores']
    nrm_scr = hist.loc[hist.labels == 0]['scores']

    # Create figure and plot the distribution.
    sns.distplot(nrm_scr, label=r'Normal Scores', bins=20)
    sns.distplot(abn_scr, label=r'Abnormal Scores', bins=20)

    plt.legend()
    plt.yticks([])
    plt.xlabel(r'Anomaly Scores')
    if options.plot_hist:
        plt.show()
    if options.save_hist:
        plt.savefig('eval_histogram.jpg')

    threshold = otsu_threshold(hist, scores)
    scores[scores >= otsu_threshold(scores)] = 1
    scores[scores <  otsu_threshold(scores)] = 0
    fpr, tpr, _ = roc_curve(labels, scores)

    print(average_precision_score(labels, scores))

def eval_net(unet, loader, device):
    """Evaluation"""
    unet.eval()
    n_val = len(loader)
    tot = 0

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                mask_pred = unet(imgs)
                tot += F.mse_loss(mask_pred, true_masks).item()
            pbar.update()

    unet.train()
    return tot / n_val

if __name__ == "__main__":
    main()
