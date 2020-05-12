import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from utils import target_peaks_gen, local_maxima, optimum, remove_outside_plot, show_res
import numpy as np


def eval_net(
    net,
    dataset,
    vis=None,
    vis_img=None,
    vis_gt=None,
    gpu=True,
    dist_peak=2,
    peak_thresh=100,
    dist_threshold=10,
    only_loss=False,
):
    criterion = nn.MSELoss()
    net.eval()
    losses = 0

    for iteration, data in enumerate(dataset):
        img = data["image"]
        target = data["gt"]
        if gpu:
            img = img.cuda()
            target = target.cuda()

        pred_img = net(img)

        loss = criterion(pred_img, target)
        losses += loss.data

        # if vis is not None:
        #     vis.images(img.cpu(), 1, win=vis_ori)
        #     vis.images(pred_img.cpu(), 1, win=vis_img)
        #     vis.close()

    return losses / iteration


def eval_net_cnn(
    net,
    dataset,
    vis=None,
    vis_img=None,
    vis_gt=None,
    gpu=True,
    dist_peak=2,
    peak_thresh=100,
    dist_threshold=10,
    only_loss=False,
):
    criterion = nn.BCELoss()
    net.eval()
    losses = 0

    for iteration, data in enumerate(dataset):
        img = data["image"]
        target = data["gt"]
        if gpu:
            img = img.cuda()
            target = target.cuda()

        pred_img = net(img)

        loss = criterion(pred_img, target)
        losses += loss.data

        # if vis is not None:
        #     vis.images(img.cpu(), 1, win=vis_ori)
        #     vis.images(pred_img.cpu(), 1, win=vis_img)
        #     vis.close()

    return losses / iteration