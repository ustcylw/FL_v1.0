#! /usr/bin/env python
# coding=utf-8
import os, sys
sys.path.insert(0, '/media/intellif/data/personal/facial_landmark/FL-py1.0')
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import cv2
from torch import nn


def get_mse(pred_points, ground_true, indices_valid=None):
    """
    :param pred_points: numpy (N,15,2)
    :param gts: numpy (N,15,2)
    :return:
    """
    pred_points = pred_points[indices_valid[0],indices_valid[1],:]
    ground_true = ground_true[indices_valid[0],indices_valid[1],:]
    pred_points = Variable(torch.from_numpy(pred_points).float(),requires_grad=False)
    ground_true = Variable(torch.from_numpy(ground_true).float(),requires_grad=False)
    criterion = nn.MSELoss()
    loss = criterion(pred_points, ground_true)
    return loss

def compute_nmse(ground_truth, predictions):
    targets = ground_truth.reshape((-1, 2))
    preds = predictions.reshape((-1, 2))
    
    N = preds.shape[0]
    L = preds.shape[1]
    rmse = np.zeros(N)
    
    for i in range(N):
        pts_pred, pts_gt = preds[i,], targets[i,]
        interocular = np.linalg.norm(pts_gt[36,]) - pts_gt[45,]
        rmse[i] = np.sum(np.linalg.norm(pts_pred - pts_gt, axis=1)) / (interocular*L)
    
    return rmse


def compute_nmse_v1(ground_truth, predictions):
    targets = ground_truth.reshape((-1, 2))
    preds = predictions.reshape((-1, 2))
    
    interocular = np.linalg.norm(targets[36,]) - targets[45,]
    rmse = np.sum(np.linalg.norm(preds - targets, axis=1)) / interocular
    
    return rmse


def mseNormlized(ground_truth, pred):
    ground_truth = ground_truth.reshape((-1, 2))
    pred = pred.reshape((-1, 2))
    eyeDistance = np.linalg.norm(ground_truth[36] - ground_truth[45])
    norm_mean = np.linalg.norm(pred - ground_truth, axis=1).sum()
    if eyeDistance > 0.0:
        return (norm_mean / eyeDistance)
    else:
        return 0.0