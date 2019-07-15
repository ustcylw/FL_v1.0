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
import models.hourglass as Hourglass
import dataset.data_set as FLDataset
import log.logger as Logger
import cv2
import loss.loss_v1 as LossV1



class Eval():
    def __init__(self, root_dir, image_list_file, params_file, use_cuda=True, save_dir='../result'):
        self.root_dir = root_dir
        self.image_list_file = image_list_file
        self.params_file = params_file
        self.use_cuda = use_cuda
        self.device = torch.device('cuda:0') if use_cuda and torch.cuda.is_available() else torch.device('cpu')
        self.save_dir = save_dir
        self.num_count = 0
        self.error_per_landmark = 0.0
        self.failure_count = 0
        self.debug  = True

        self.image_list = self.load_data_file()
        if '' != self.params_file:
            self.model = self.load_model()
        
    def load_model(self):
        print('loading model ... {}'.format(self.params_file))
        net = Hourglass.HourglassNetV1(
            in_channels=32,
            channels=64,
            hourglass_multiplier=1.0,
            bottleneck_multiplier=1.0,
            num_order=3,
            channels_width=128
        )
        net.load_state_dict(torch.load(self.params_file))

        net.float().to(self.device)
        net.eval()
        print('load model complete')
        return net

    def reset_model(self, params_file):
        print('reset model ...')
        self.params_file = params_file
        self.model = self.load_model()
        print('reset model complete.')

    def reset_model_ex(self, params_file, load_model_func, **kwargs):
        pass

    def get_peak_points(self, heatmaps):
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

    def calculate_mask(self, heatmaps_targets):
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

    def get_image_list(self):
        part_names = os.listdir(self.root_dir)
        image_list = []
        for part_name in part_names:
            if part_name in ['01_Indoor', '02_Outdoor']:
                continue
            part_path = os.path.join(self.root_dir, part_name)
            files = os.listdir(part_path)
            for f in files:
                if f.endswith('png'):
                    image_file = os.path.join(part_path, f)
                    point_file = image_file.replace('png', 'pts')
                    if os.path.exists(point_file):
                        image_list.append(os.path.join(part_name, f))
        return image_list

    def load_data_file(self):
        print('loading data file ... {}'.format(self.image_list_file))
        self.image_list = []
        if not os.path.exists(self.image_list_file):
            self.image_list = self.get_image_list()
            image_str = ''
            for image in self.image_list:
                image_str += image
                image_str += '\n'
            with open(self.image_list_file, 'w+') as df:
                df.write(image_str)
                df.flush()
        else:
            lines = None
            with open(self.image_list_file, 'r') as lf:
                lines = lf.readlines()
            self.image_list = [line[0:-1] for line in lines]
        print('load data file complete')

    def eval(self):
        
        all_result = []
        
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        for i, image_name in enumerate(self.image_list):
            # print('[{}/{}] dealing with {}'.format(i,len(self.image_list), image_name))
            image_file = os.path.join(self.root_dir, image_name)
            image = cv2.imread(image_file)
            image_ = image.copy()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            image = image.transpose((2, 0, 1))
            image = (image-127.5)/128.0
            image = image[np.newaxis, :, :, :]
            images = torch.Tensor(image).float().to(self.device)
            lines = None
            point_file = image_file.replace('png', 'pts')
            with open(point_file, 'r') as lf:
                lines = lf.readlines()
            points = lines[0].split('\t')
            points = np.array([int(round(float(item))) for item in points])
            points = points.reshape((-1, 2))
            pred_heatmaps = self.model.forward(images)

            """
            可视化预测结果
            demo_img = images[0].cpu().data.numpy()[0]
            demo_img = (demo_img * 255.).astype(np.uint8)
            demo_heatmaps = pred_heatmaps[0].cpu().data.numpy()[np.newaxis,...]
            demo_pred_poins = get_peak_points(demo_heatmaps)[0] # (15,2)
            plt.imshow(demo_img,cmap='gray')
            plt.scatter(demo_pred_poins[:,0],demo_pred_poins[:,1])
            plt.show()
            """

            pred_points = self.get_peak_points(pred_heatmaps.cpu().data.numpy()) #(N,15,2)
            # pred_points = pred_points.reshape((pred_points.shape[0],-1)) #(N,30)
            pred_points = pred_points.reshape((-1, 2))

            # # 筛选出要查询的features
            # for idx,img_id in enumerate(ids):
            #     result_img = lookup_df[lookup_df['ImageId'] == img_id]
            #     # 映射feature names to ids
            #     fea_names = result_img['FeatureName'].as_matrix()
            #     fea_ids = [config['featurename2id'][name] for name in fea_names]
            #     pred_values = pred_points[idx][fea_ids]
            #     result_img['Location'] = pred_values
            #     all_result.append(result_img)

            font = cv2.FONT_HERSHEY_SIMPLEX
            for i in range(pred_points.shape[0]):
                x1, y1 = int(round(pred_points[i][0])), int(round(pred_points[i][1]))
                cv2.circle(image_, (x1, y1), 1, (0, 0, 255), 1)
                if True:
                    x2, y2 = int(round(points[i][0])), int(round(points[i][1]))
                    cv2.circle(image_, (x2, y2), 1, (0, 255, 0), 1)
                if False:
                    cv2.putText(image_, str(i), (x2, y2), font, 0.1, (255, 0, 0), 0.1, cv2.LINE_4)
                if True:
                    cv2.line(image_, (x1, y1), (x2, y2), (255, 255, 0))
            image_ = cv2.resize(image_, (512, 512))
            save_file = os.path.join(self.save_dir, image_name.replace('/', '_'))
            cv2.imwrite(save_file, image_)

            if True:
                cv2.imshow('image', image_)
                if cv2.waitKey(100) == ord('q'):
                    sys.exit(0)

            normlized = LossV1.mseNormlized(points, pred_points)
            # print('normlized: ', normlized, image_file)
            self.error_per_landmark += normlized
            # print('error_per_landmark: ', self.error_per_landmark)
            self.num_count += 1
            if normlized > 6.8: # 0.1
                self.failure_count += 1
                if self.debug:
                    failure_file = os.path.join('../err', image_name.replace('/', '_'))
                    cv2.imwrite(failure_file, image_)
    
    def get_result(self):
        self.mean_error = self.error_per_landmark / self.num_count
        print('\nResult: \nmean error: %.05f \n failures(err>0.1): %.02f%%(%d/%d)'%(
            self.mean_error, 
            (self.failure_count/float(self.num_count)*100.0), 
            self.failure_count, 
            self.num_count
        ))
        return self.mean_error



if __name__ == '__main__':
    
    root_dir = '/media/intellif/data/datasets/300w/300w'
    image_list_file = '/media/intellif/data/datasets/300w/300w/image_list.txt'
    save_dir = '../result'
    params_list_file = [
        './../params/FL_315_58160_model.ckpt'
    ]

    use_cuda = True
    
    
    eval = Eval(
        root_dir=root_dir,
        image_list_file=image_list_file,
        params_file='',
        use_cuda=use_cuda,
        save_dir=save_dir
    )
    eval.load_data_file()
    
    for idx, params_file in enumerate(params_list_file):
        print('='*80)
        print('\n[{}/{}]'.format(len(params_list_file), idx))
        eval.reset_model(params_file=params_file)
        eval.eval()
        eval.get_result()
    
    


