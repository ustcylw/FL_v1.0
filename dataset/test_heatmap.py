#! coding: utf-8
import numpy as np

def putGaussianMap(center, visible_flag, crop_size_y, crop_size_x, stride, sigma):
    grid_y = int(round(crop_size_y / stride))
    grid_x = int(round(crop_size_x / stride))
    if visible_flag == False:
        return np.zeros((grid_y,grid_x))
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(grid_y)]
    x_range = [i for i in range(grid_x)]
    xx, yy = np.meshgrid(x_range, y_range)
    # print('xx-yy: ', xx.shape, yy.shape)
    # print('xx0: ', xx)
    xx = xx * stride + start
    yy = yy * stride + start
    # print('xx1: ', xx, center, sigma)
    # print('xx - center[0]: ', xx - center[0])
    # print('(xx - center[0]) ** 2: ', (xx - center[0]) ** 2)
    # print('yy - center[1]: ', yy - center[1])
    # print('(yy - center[1]) ** 2: ', (yy - center[1]) ** 2)
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    # print('d2: ', d2.shape, d2)
    exponent = d2 / 2.0 / sigma / sigma
    # print('exponent: ', exponent.shape, np.max(exponent), exponent, sigma)
    heatmap = np.exp(-exponent)
    # print('heatmap: ', heatmap.shape, np.max(heatmap))
    return heatmap



if __name__ == '__main__':

    center = (5, 5)
    x, y = 128, 128
    stride = 1.0
    sigma = 2.0

    # heatmap = putGaussianMap(center, True, x, y, stride, sigma)
    # print('[***] heatmap: ', heatmap.shape, heatmap)
    heatmap = putGaussianMap(center, True, x, y, stride, sigma*10000)
    print('[***] heatmap: ', heatmap.shape, np.max(heatmap), heatmap)
    