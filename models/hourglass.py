#! /usr/bin/env python
# coding=utf-8
""" pytorch实现：stacked hourglass network architecture"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.nn import UpsamplingNearest2d,Upsample
from torch.autograd import Variable





class BottleNetBlock(nn.Module):
    def __init__(self, in_channels, channels, multiplier, **kwargs): 
        super(BottleNetBlock, self).__init__(**kwargs)
        
        inner_channels = int(in_channels * multiplier)
        
        self.bn1 = torch.nn.BatchNorm2d(in_channels)
        self.act1 = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=inner_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False
        )
        self.bn2 = torch.nn.BatchNorm2d(inner_channels)
        self.act2 = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv2d(
            in_channels=inner_channels,
            out_channels=inner_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            dilation=1,
            groups=1,
            bias=False
        )
        self.bn3 = torch.nn.BatchNorm2d(inner_channels)
        self.act3 = torch.nn.ReLU()
        self.conv3 = torch.nn.Conv2d(
            in_channels=inner_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False
        )
        self.shortcut = torch.nn.Conv2d(
            in_channels=in_channels,
            out_channels=channels,
            kernel_size=1,
            stride=1,
            padding=0,
            dilation=1,
            groups=1,
            bias=False
        )
    
    def forward(self, inputs):
        residual = self.conv1(self.act1(self.bn1(inputs)))
        residual = self.conv2(self.act2(self.bn2(residual)))
        residual = self.conv3(self.act3(self.bn3(residual)))
        shortcut = self.shortcut(inputs)
        return residual + shortcut


# class BottleNetGroup(tf.keras.Model):
class BottleNetGroup(torch.nn.Module):
    def __init__(self, in_channels, channels, hourglass_multiplier, bottleneck_multiplier, n, **kwargs): 
        super(BottleNetGroup, self).__init__(**kwargs)
        
        inner_channels = int(in_channels * hourglass_multiplier)
        
        self.chain_layers = torch.nn.Sequential()
        self.chain_layers.add_module(
            'BottleNetBlock', BottleNetBlock(
                in_channels,
                inner_channels,
                bottleneck_multiplier
            )
        )
        for i in range(n):
            self.chain_layers.add_module(
                'BottleNetBlock', BottleNetBlock(
                    inner_channels,
                    inner_channels,
                    bottleneck_multiplier
                )
            )
        self.chain_layers.add_module(
            'BottleNetBlock', BottleNetBlock(
                inner_channels,
                channels,
                bottleneck_multiplier
            )
        )
        
    def forward(self, inputs):
        return self.chain_layers(inputs)


# class DownSampling(tf.keras.Model):
class DownSampling(torch.nn.Module):
    def __init__(self, scale, method, channels):
        super(DownSampling, self).__init__()
        self.scale_num = int(scale/2)
        self.method = method
        self.down_sampling = torch.nn.Sequential()
        for i in range(self.scale_num):
            if 'max_pooling' == self.method:
                self.down_sampling.add_module(
                    'MaxPool2D', torch.nn.MaxPool2d(2, 2)
                )
            elif 'avg_pooling' == self.method:
                self.down_sampling.add_module(
                    'AvgPool2D', torch.nn.AvgPool2d(2, 2)
                )
            else:
                self.down_sampling.add_module(
                    'Conv2D', torch.nn.Conv2d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=3,
                        stride=2
                    )
                )

    def forward(self, inputs):
        return self.down_sampling(inputs)


# class UpSampling(tf.keras.Model):
class UpSampling(torch.nn.Module):
    def __init__(self, in_channels, scale, method='up_sampling', interpolation='nearest'):
        super(UpSampling, self).__init__()
        self.scale_num = scale
        self.method = method
        self.interpolation = interpolation
        self.up_sampling = torch.nn.Sequential()
        for i in range(self.scale_num):
            if 'up_sampling' == self.method:
                self.up_sampling.add_module(
                    'UpsamplingBilinear2D', torch.nn.UpsamplingBilinear2d(scale_factor=2)
                )
            else:
                self.up_sampling.add_module(
                    'ConvTranspose2d', torch.nn.ConvTranspose2d(
                        in_channels=in_channels,
                        out_channels=in_channels,
                        kernel_size=3,
                        stride=1
                    )
                )

    def forward(self, inputs):
        return self.up_sampling(inputs)


# class HourglassBlock(tf.keras.Model):
class HourglassBlock(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        hourglass_multiplier,
        bottleneck_multiplier,
        sub_block=None,
        **kwargs
    ):
        super(HourglassBlock, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.channels = channels
        self.bottleneck_multiplier = bottleneck_multiplier
        self.hourglass_multiplier = hourglass_multiplier
        
        self.shortcut = BottleNetGroup(
            self.in_channels,
            self.channels,
            self.hourglass_multiplier,
            self.bottleneck_multiplier,
            1
        )
        self.down_sampling = DownSampling(
            2, 'max_pooling',
            self.in_channels
        )
        self.feature_chain = BottleNetGroup(
            self.in_channels,
            self.in_channels,
            self.hourglass_multiplier,
            self.bottleneck_multiplier,
            1
        )
        if sub_block:
            self.sub_block = sub_block
        else:
            self.sub_block = BottleNetBlock(
                self.in_channels,
                self.channels,
                self.bottleneck_multiplier
            )
        self.adjust_block = BottleNetBlock(
            self.channels,
            self.channels,
            self.bottleneck_multiplier
        )
        self.up_sampling = UpSampling(
            in_channels=0,
            scale=1
        )

    def forward(self, inputs):
        shortcut = self.shortcut(inputs)
        down_sampling = self.down_sampling(inputs)
        chain = self.feature_chain(down_sampling)
        sub_block = self.sub_block(chain)
        adjust_block = self.adjust_block(sub_block)
        up_sampling = self.up_sampling(adjust_block)
        return shortcut + up_sampling

class HourglassNetV1(torch.nn.Module):
# class HourglassNetV1(tf.keras.layers.Layer):
    def __init__(
        self,
        in_channels,
        channels,
        hourglass_multiplier,
        bottleneck_multiplier,
        num_order=1,
        channels_width=128,
        **kwargs
    ):
        super(HourglassNetV1, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.channels = channels
        self.hourglass_multiplier = hourglass_multiplier
        self.bottleneck_multiplier = bottleneck_multiplier
        self.num_order = num_order
        self.channels_width = channels_width

        self.in_block = torch.nn.Sequential()
        self.in_block.add_module(
            'Conv2d', torch.nn.Conv2d(
                in_channels=3,
                out_channels=64,
                kernel_size=3,
                stride=1,
                padding=1
            )
        )
        self.in_block.add_module(
            'BottleNetGroup', BottleNetGroup(
                in_channels=64,
                channels=self.in_channels,
                hourglass_multiplier=1.0,
                bottleneck_multiplier=1.0,
                n=1
            )
        )
        self.n_orders = torch.nn.Sequential()
        first_order = HourglassBlock(
            in_channels=self.in_channels,
            channels=self.channels,
            hourglass_multiplier=1.0,
            bottleneck_multiplier=1.0,
            sub_block=None
        )
        up_order = first_order
        for i in range(self.num_order):
            cur_order = HourglassBlock(
                in_channels=self.in_channels,
                channels=self.channels,
                hourglass_multiplier=1.0,
                bottleneck_multiplier=1.0,
                sub_block=up_order
            )
            up_order = cur_order
        self.n_orders.add_module('up_order', up_order)
        
        self.heatmap = torch.nn.Sequential()
        self.heatmap.add_module(
            'BottleNetBlock', BottleNetBlock(
                in_channels=self.channels,
                channels=self.in_channels,
                multiplier=1.0
            )
        )
        self.heatmap.add_module(
            'Conv2D', torch.nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=68,
                kernel_size=1,
                stride=1,
                bias=False
            )
        )
        self.heatmap.add_module(
            'Sigmoid', torch.nn.Sigmoid()            
        )

    def forward(self, inputs):
        in_block = self.in_block(inputs)
        n_order = self.n_orders(in_block)
        heatmap = self.heatmap(n_order)
        return heatmap






if __name__ == '__main__':
    
    # data = tf.Variable(np.linspace(0, 1, 256*256*3).reshape(1, 256, 256, 3).astype(np.float))
    data = torch.Tensor(np.linspace(0, 1, 128*128*3).reshape(1, 3, 128, 128).astype(np.float))
    print('data: ', data.shape)
    
    net = HourglassNetV1(
        in_channels=64,
        channels=128,
        hourglass_multiplier=1.0,
        bottleneck_multiplier=1.0,
        num_order=4,
        channels_width=128
    )
    print('='*80)
    print('net: ', net)
    print('='*80)
    
    # net.fit(data, tf.Variable(np.array([0])), epochs=3)
    
    pred = net(data)
    print('pred: ', pred.shape)









# if __name__ == '__main__':
#     from torch.nn import MSELoss
#     critical = MSELoss()

#     dataset = tempDataset()
#     dataLoader = DataLoader(dataset=dataset)
#     shg = StackedHourGlass()
#     optimizer = optim.SGD(shg.parameters(), lr=0.01, momentum=0.9,weight_decay=1e-4)
#     for i,(x,y) in enumerate(dataLoader):
#         x = Variable(x,requires_grad=True).float()
#         y = Variable(y).float()
#         y_pred = shg.forward(x)
#         loss = critical(y_pred[0],y[0])
#         print('loss : {}'.format(loss))
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()