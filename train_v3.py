#! /usr/bin/env python
# coding=utf-8
import os, sys
import torch
from torch.autograd import Variable
from torch.backends import cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pprint
import models.hourglass as Hourglass
import dataset.data_set as FLDataset
from tensorboardX import SummaryWriter
import log.logger as Logger
import cv2
import time


config = dict()
config['lr'] = 0.00001
config['momentum'] = 0.9
config['weight_decay'] = 1e-4
config['start_epoch'] = 200
config['epoch_num'] = 100
config['batch_size'] = 60  # 128
config['sigma'] = 5.
config['debug_vis'] = False         # 是否可视化heatmaps
config['is_test'] = False
config['save_freq'] = 1
config['save_dir'] = './params'
config['pre-trained'] = './params/FL_102_20442_model.ckpt'  # params/FL_9_400_model.ckpt'
config['eval_freq'] = 5
config['debug'] = False
config['log_dir'] = './logs'
config['gpu_ids'] = '0'
config['no_cuda'] = False
config['use_gpu'] = not config['no_cuda'] and torch.cuda.is_available()
config['featurename2id'] = {
    'left_eye_center_x':0,
    'left_eye_center_y':1,
    'right_eye_center_x':2,
    'right_eye_center_y':3,
    'left_eye_inner_corner_x':4,
    'left_eye_inner_corner_y':5,
    'left_eye_outer_corner_x':6,
    'left_eye_outer_corner_y':7,
    'right_eye_inner_corner_x':8,
    'right_eye_inner_corner_y':9,
    'right_eye_outer_corner_x':10,
    'right_eye_outer_corner_y':11,
    'left_eyebrow_inner_end_x':12,
    'left_eyebrow_inner_end_y':13,
    'left_eyebrow_outer_end_x':14,
    'left_eyebrow_outer_end_y':15,
    'right_eyebrow_inner_end_x':16,
    'right_eyebrow_inner_end_y':17,
    'right_eyebrow_outer_end_x':18,
    'right_eyebrow_outer_end_y':19,
    'nose_tip_x':20,
    'nose_tip_y':21,
    'mouth_left_corner_x':22,
    'mouth_left_corner_y':23,
    'mouth_right_corner_x':24,
    'mouth_right_corner_y':25,
    'mouth_center_top_lip_x':26,
    'mouth_center_top_lip_y':27,
    'mouth_center_bottom_lip_x':28,
    'mouth_center_bottom_lip_y':29,
}


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

def get_mse(pred_points, ground_true, indices_valid=None):
    """
    :param pred_points: numpy (N,15,2)
    :param gts: numpy (N,15,2)
    :return:
    """
    # print('[***] predicts: ', pred_points.shape, pred_points)
    # print('[***] ground_true: ', ground_true.shape, ground_true)
    pred_points = pred_points[indices_valid[0],indices_valid[1],:]
    ground_true = ground_true[indices_valid[0],indices_valid[1],:]
    pred_points = Variable(torch.from_numpy(pred_points).float(),requires_grad=False)
    ground_true = Variable(torch.from_numpy(ground_true).float(),requires_grad=False)
    criterion = nn.MSELoss()
    loss = criterion(pred_points, ground_true)
    # print('[***] predicts: ', pred_points.shape, pred_points)
    # print('[***] ground_true: ', ground_true.shape, ground_true)
    # print('[***] loss: ', loss.shape, loss  )
    return loss

def calculate_mask(heatmaps_target):
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

if __name__ == '__main__':

    # root_dir = '/media/intellif/data/datasets/300VW/test'
    # save_file = '/media/intellif/data/datasets/300VW/test_1.txt'
    root_dir = '/media/intellif/data/datasets/300VW/300VW_Dataset_2015_12_14'
    save_file = '/media/intellif/data/datasets/300VW/300VW_Dataset.txt'

    # log
    logger = Logger.LogHandler('train_002')

    # summary-writer
    time_dir = time.strftime('%Y-%m-%d--%H-%M-%S',time.localtime(time.time()))
    log_dir = os.path.join(config['log_dir'], time_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    # load net-structure
    pprint.pprint(config)
    torch.manual_seed(0)
    cudnn.benchmark = True
    net = Hourglass.HourglassNetV1(
        in_channels=32,
        channels=64,
        hourglass_multiplier=1.0,
        bottleneck_multiplier=1.0,
        num_order=3,
        channels_width=128
    )
    print('='*80)
    print('net: ', net)
    print('='*80)
    with writer:
        writer.add_graph(net, torch.rand([1, 3, 128, 128]), True)
    net.float().cuda()
    net.train()
    
    # loss
    criterion = nn.MSELoss()

    # optimizer
    # optimizer = optim.SGD(net.parameters(), lr=config['lr'], momentum=config['momentum'] , weight_decay=config['weight_decay'])
    optimizer = optim.Adam(net.parameters(),lr=config['lr'])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [2, 6], 0.1)
    
    # load dataset
    trainDataset = FLDataset.FLDatasetV1(root_dir=root_dir, dataset_list=save_file, batch_size=config['batch_size'])
    trainDataLoader = DataLoader(dataset=trainDataset,batch_size=config['batch_size'],shuffle=True)
    sample_num = len(trainDataset)

    if (config['pre-trained'] != ''):
        print('loading pre-trained model ... {}'.format(config['pre-trained']))
        net.load_state_dict(torch.load(config['pre-trained']))
        print('load pre-trained model complete')

    step = 0
    for epoch in range(config['start_epoch'], config['epoch_num']+config['start_epoch']):
        lr_scheduler.step()
        running_loss = 0.0
        for idx, (inputs, heatmaps_targets, gts) in enumerate(trainDataLoader):
            inputs = Variable(inputs).cuda()
            heatmaps_targets = Variable(heatmaps_targets).cuda()
            mask,indices_valid = calculate_mask(heatmaps_targets)
            
            optimizer.zero_grad()
            outputs = net(inputs)  # [B, 68, 128, 128]
            outputs = outputs * mask
            heatmaps_targets = heatmaps_targets * mask
            loss = criterion(outputs, heatmaps_targets)
            loss.backward()
            optimizer.step()

            # 统计最大值与最小值
            v_max = torch.max(outputs)
            v_min = torch.min(outputs)

            # 评估
            all_peak_points = get_peak_points(heatmaps_targets.cpu().data.numpy())
            loss_coor = get_mse(all_peak_points, gts.numpy(),indices_valid)

            logger.info('  [{}/{}/{}/{}] loss:{:12} loss_coor: {:12} max: {:10} min : {} lr: {}'.format(
                epoch, step, sample_num, idx*config['batch_size'],
                loss.data.item(), loss_coor.data.item(),
                v_max.data.item(), v_min.data.item(),
                lr_scheduler.get_lr()[0]
            ))

            step += 1
            
            # summary-writer loss and weigth
            writer.add_scalar('loss', loss.data.item(), step)
            writer.add_scalar('coor-loss', loss_coor.data.item(), step)
            for k, (name, param) in enumerate(net.named_parameters()):
                if 'bn' not in name:
                    writer.add_histogram(name, param.clone().cpu().data.numpy(), step)
            # for jj in range(outputs.shape[0]):
            #     img = (inputs[jj,:,:,:].clone().cpu().data.numpy()*128+127.5).astype(np.uint8)
            #     img = img.transpose((1,2,0))
            #     pts = all_peak_points[jj]#*128
            #     # print('pts: ', pts)
            #     print('img: ', img.shape, img)
            #     for kk in range(68):
            #         cv2.circle(img, (int(round(pts[kk,0])), int(round(pts[kk,1]))), 3, (0, 255, 0), 3)
            #         print('pts: ', pts[kk,:])
            #     img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            #     cv2.imshow('123', img)
            #     if cv2.waitKey(0) == ord('q'):
            #         sys.exit(0)
            #     img = img.transpose((2,0,1))
            #     # print('outputs[0][jj]: ', outputs[0][jj].clone().cpu().data.numpy().reshape((1, 128, 128)).shape)
            #     writer.add_image(str(jj), img.reshape((3, 128, 128)), step)

            
        # save params
        if (epoch+1) % config['save_freq'] == 0 or epoch == config['epoch_num'] - 1:
            save_file = config['save_dir'] + '/FL_{}_{}_model.ckpt'.format(epoch, step)
            logger.info('saveing model {}'.format(save_file))
            torch.save(net.state_dict(), save_file)
            logger.info('save model complete.')

    wirter.close()