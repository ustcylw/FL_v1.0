#! /usr/bin/env python
# coding=utf-8
import os, sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset,DataLoader
import copy
# import shuffle
import matplotlib.pyplot as plt
import cv2



def get_sigma_mask():
    mask = np.zeros((68,))
    mask[0:17] = 1.6  # contour
    mask[17:27] = 1.0  # eyebow
    mask[27:31] = 1.2  # nose-1
    mask[31:36] = 1.0  # nose-2
    mask[36:48] = 1.0  # eyes
    mask[48:68] = 1.0  # mouse
    return mask


def plot_sample(x, y, axis):
    """

    :param x: (9216,)
    :param y: (15,2)
    :param axis:
    :return:
    """
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[:,0], y[:,1], marker='x', s=10)

def plot_demo(X,y):
    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(
        left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(X[i], y[i], ax)

    plt.show()



class FLDatasetV1(Dataset):
    '''
    for landmark
    '''
    def __init__(self, root_dir, dataset_list, batch_size):
        super(FLDatasetV1, self).__init__()
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.dataset_list = dataset_list
        self.image_list = []
        self.label_list = []
        self.__make_dataset_list()
        self.__parse_dataset_list()
        self.__shape = (128, 128, 3)
        self.__is_test = False
        self.__debug_vis = False
        self.__sigma = 5
        self.__sigma_mask = get_sigma_mask()
        self.__stride = 1
                
    def __make_dataset_list(self): 
        if not os.path.exists(self.dataset_list):
            print('creating dataset list ...')
            """创建训练集tfrecord""" 
            NUM = 1 # 显示创建过程（计数） 
            ids = os.listdir(self.root_dir)
            lines = ''
            for idx1, id in enumerate(ids): 
                sub_path = os.path.join(self.root_dir, id)
                sub_path = os.path.join(sub_path, 'crop_128_128')
                items = os.listdir(sub_path)
                for item in items: 
                    if not item.endswith('jpg'):
                        continue
                    print('dealing with {}'.format(item ))
                    img_path = os.path.join(sub_path, item)
                    pts_path = img_path.replace('jpg', 'pts')
                    if not os.path.exists(pts_path):
                        print('points file is not exist !!! {}'.format(pts_path))
                        continue
                    lines += img_path
                    lines += ','
                    lines += pts_path
                    lines += '\n'
                    self.image_list.append(img_path)
                    self.label_list.append(pts_path)
                    
                    print('Creating train record in ',NUM) 
                    NUM += 1 
            with open(self.dataset_list, 'w+') as sf:
                sf.write(lines)
                sf.flush()
                
            print("Create dataset list successful!")
        else:
            print('dataset list is already exit.')

    def __parse_dataset_list(self):
        if (len(self.image_list) == 0) or (len(self.label_list) == 0):
            print('parsing dataset list ...')
            self.image_list = []
            self.label_list = []
            lines = ''
            with open(self.dataset_list, 'r') as df:
                lines = df.readlines()
            for line in lines:
                image_file, label_file = line.split(',')
                self.image_list.append(image_file)
                self.label_list.append(label_file[0:-1])
            print('parse dataset list successful.')

    def __generator(self):
        while self.__current_corsor+self.batch_size < self.__max_corsor:
            yield (
                self.image_list[self.__current_corsor:self.batch_size],
                self.label_list[self.__current_corsor:self.batch_size]
            )
            self.__current_corsor += self.__current_corsor + self.batch_size

    def __parse_function(self, image_file, label_file):
        landmarks = None
        with open(label_file, 'r') as sf:
            landmarks = sf.readlines()
        landmarks = landmarks[0].split('\t')
        landmarks = np.array([int(round(float(i))) for i in landmarks]).reshape(68, 2)
        image = cv2.imread(image_file)
        if image is None:
            print('image is None !!! {}'.format(image_file))
        image = cv2.resize(image, (self.__shape[0], self.__shape[1])) # resize图片大小
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))  # swapaxes(1,2).swapaxes(2,1)
        # print('image: ', image.shape, landmarks.shape)
        return image, landmarks

    def __getitem__(self, item):
        H,W = self.__shape[0], self.__shape[1]
        image_file = self.image_list[item]
        point_file = self.label_list[item]
        
        
        image, label = self.__parse_function(image_file, point_file)
        if self.__is_test:
            return image, label

        heatmaps = self.__putGaussianMaps(label, H, W, self.__stride, self.__sigma)

        if self.__debug_vis:
            for i in range(heatmaps.shape[0]):
                # img = copy.deepcopy(image).astype(np.uint8).reshape((H,W, 3))
                img = image.copy()
                img = np.transpose(img, (1,2,0))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.uint8)
                self.visualize_heatmap_target(img,copy.deepcopy(heatmaps[i]),1)

        image = image.astype(np.float32)
        image = (image-127.5) / 128.0
        heatmaps = heatmaps.astype(np.float32)
        
        return image, heatmaps, label

    def __putGaussianMap(self, center, visible_flag, crop_size_y, crop_size_x, stride, sigma):
        """
        根据一个中心点,生成一个heatmap
        :param center:
        :return:
        """
        grid_y = int(round(crop_size_y / stride))
        grid_x = int(round(crop_size_x / stride))
        if visible_flag == False:
            return np.zeros((grid_y,grid_x))
        start = stride / 2.0 - 0.5
        y_range = [i for i in range(grid_y)]
        x_range = [i for i in range(grid_x)]
        xx, yy = np.meshgrid(x_range, y_range)
        xx = xx * stride + start
        yy = yy * stride + start
        d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
        exponent = d2 / 2.0 / sigma / sigma
        heatmap = np.exp(-exponent)
        return heatmap

    def __putGaussianMaps(self,keypoints,crop_size_y, crop_size_x, stride, sigma):
        """

        :param keypoints: (15,2)
        :param crop_size_y: int
        :param crop_size_x: int
        :param stride: int
        :param sigma: float
        :return:
        """
        all_keypoints = keypoints
        point_num = all_keypoints.shape[0]
        heatmaps_this_img = []
        for k in range(point_num):
            flag = ~np.isnan(all_keypoints[k,0])
            heatmap = self.__putGaussianMap(all_keypoints[k],flag,crop_size_y,crop_size_x,stride,sigma*self.__sigma_mask[k])
            heatmap = heatmap[np.newaxis,...]
            heatmaps_this_img.append(heatmap)
        heatmaps_this_img = np.concatenate(heatmaps_this_img,axis=0) # (num_joint,crop_size_y/stride,crop_size_x/stride)
        return heatmaps_this_img

    def visualize_heatmap_target(self,oriImg,heatmap,stride):
        plt.imshow(oriImg)
        plt.imshow(heatmap, alpha=.5)
        plt.show()

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':

    root_dir = '/media/intellif/data/datasets/300VW/test'
    save_file = '/media/intellif/data/datasets/300VW/test_1.txt'
    batch_size = 128
    
    dataset = FLDatasetV1(root_dir=root_dir, dataset_list=save_file, batch_size=batch_size)
    dataLoader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=False)
    for e in range(3):
        for i, (x, y ,gt) in enumerate(dataLoader):
            print(e, i, x.size())
            print(e, i, y.size())
            print(e, i, gt.size())
