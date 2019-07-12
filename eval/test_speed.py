#! /usr/bin/env python
# coding=utf-8
import os, sys
sys.path.insert(0, '/media/intellif/data/personal/facial_landmark/FL-py1.0')
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
import matplotlib.pyplot as plt
import pandas as pd
import time
import models.hourglass as Hourglass
import dataset.data_set as FLDataset
import log.logger as Logger
import cv2


def get_peak_points(heatmaps):
    """
    :param heatmaps: numpy array (N,15,96,96)
    :return:numpy array (N,15,2)
    """
    N,C,H,W = heatmaps.shape
    all_peak_points = []
    for i in range(N):
        peak_points = []
        for j in range(C):
            yy,xx = np.where(heatmaps[i,j] == heatmaps[i,j].max())
            y = yy[0]
            x = xx[0]
            peak_points.append([x,y])
        all_peak_points.append(peak_points)
    all_peak_points = np.array(all_peak_points)
    return all_peak_points


def calculate_mask(heatmaps_targets):
    """

    :param heatmaps_target: Variable (N,15,96,96)
    :return: Variable (N,15,96,96)
    """
    N,C,_,_ = heatmaps_targets.size()
    N_idx = []
    C_idx = []
    for n in range(N):
        for c in range(C):
            max_v = heatmaps_targets[n,c,:,:].max().data.item()
            if max_v != 0.0:
                N_idx.append(n)
                C_idx.append(c)
    mask = Variable(torch.zeros(heatmaps_targets.size()))
    mask[N_idx,C_idx,:,:] = 1.
    mask = mask.float().cuda()
    return mask,[N_idx,C_idx]


def get_image_list(root_dir):
    part_names = os.listdir(root_dir)
    image_list = []
    for part_name in part_names:
        if part_name in ['01_Indoor', '02_Outdoor']:
            continue
        part_path = os.path.join(root_dir, part_name)
        files = os.listdir(part_path)
        for f in files:
            if f.endswith('png'):
                image_file = os.path.join(part_path, f)
                point_file = image_file.replace('png', 'pts')
                if os.path.exists(point_file):
                    image_list.append(os.path.join(part_name, f))
    return image_list


def test():
    
    root_dir = '/media/intellif/data/datasets/300w/300w'
    image_list_file = '/media/intellif/data/datasets/300w/300w/image_list.txt'
    save_dir = '../result'
    pre_trained = './../params/FL_1_13628_model.ckpt'
    
    use_cuda = False
    device = torch.device('cuda:0') if use_cuda and torch.cuda.is_available() else torch.device('cpu')

    # 加载模型
    print('loading model ...')
    net = Hourglass.HourglassNetV1(
        in_channels=32,
        channels=64,
        hourglass_multiplier=1.0,
        bottleneck_multiplier=1.0,
        num_order=3,
        channels_width=128
    )
    net.load_state_dict(torch.load(pre_trained))

    net.float().to(device)
    net.eval()
    print('load model complete')
 
    print('loading data file ... {}'.format(image_list_file))
    image_list = None
    if not os.path.exists(image_list_file):
        image_list = get_image_list(root_dir)
        image_str = ''
        for image in image_list:
            image_str += image
            image_str += '\n'
        with open(image_list_file, 'w+') as df:
            df.write(image_str)
            df.flush()
    else:
        image_list = None
        with open(image_list_file, 'r') as lf:
            image_list = lf.readlines()
        image_list = [item[0:-1] for item in image_list]
    print('load data file complete')

    all_result = []
    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_count = 0
    error_per_landmark = 0.0
    failure_count = 0
    debug  = True
    start = time.time()
    for i, image_name in enumerate(image_list):
        print('[{}/{}] dealing with {}'.format(i,len(image_list), image_name))
        image_file = os.path.join(root_dir, image_name)
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image.transpose((2, 0, 1))
        image = (image-127.5)/128.0
        image = image[np.newaxis, :, :, :]
        images = torch.Tensor(image).float().to(device)
        pred_heatmaps = net.forward(images)
        pred_points = get_peak_points(pred_heatmaps.cpu().data.numpy()) #(N,15,2)

        num_count += 1

    end = time.time()
    speed = (end - start)/len(image_list)
    
    mean_error = error_per_landmark / num_count
    print('\n speed: {} {}'.format(speed, end-start))









    #     # loss = get_mse(demo_pred_poins[np.newaxis,...],gts)
    # result_df = pd.concat(all_result)
    # result_df = result_df.drop(columns=['ImageId','FeatureName'])
    # result_df.to_csv('data/result_909.csv',index=False)

if __name__ == '__main__':
    test()