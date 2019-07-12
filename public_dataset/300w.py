#! /usr/bin/env python
# coding: utf-8
import os, sys
import cv2
import numpy as np
import math



def adjust_bbox(bbox, shape):
    scale = 8
    w_bbox = bbox[1][0]-bbox[0][0]
    h_bbox = bbox[1][1] - bbox[0][1]
    semi_expand_edge = int(abs(w_bbox-h_bbox)/2)
    edge_max = w_bbox if w_bbox > h_bbox else h_bbox
    if w_bbox >h_bbox:
        cube_bbox = np.array([
            [bbox[0][0], bbox[0][1]-semi_expand_edge],
            [bbox[0][0]+edge_max, bbox[0][1]+edge_max-semi_expand_edge]
        ])
    else:
        cube_bbox = np.array([
            [bbox[0][0]-semi_expand_edge, bbox[0][1]],
            [bbox[0][0]+edge_max-semi_expand_edge, bbox[0][1]+edge_max],
        ])
    expand_edge = edge_max/scale
    semi_expand_edge = int(expand_edge/2)
    cube_bbox = np.array([
        [cube_bbox[0][0]-semi_expand_edge, cube_bbox[0][1]-semi_expand_edge],
        [cube_bbox[1][0]+semi_expand_edge, cube_bbox[1][1]+semi_expand_edge]
    ])
    return cube_bbox

def draw_points(image, points, color=(0,0,255)):
    for point in points:
        cv2.circle(image, (point[0], point[1]), 2, color, 2)

def draw_box(image, bbox, color=(0,0,255)):
    cv2.rectangle(image, (bbox[0][0], bbox[0][1]), (bbox[1][0], bbox[1][1]), color)

def crop(root_dir):
    video_dir_names = os.listdir(root_dir)
    for idx1, video_dir_name in enumerate(video_dir_names):
        video_path = os.path.join(root_dir, video_dir_name)
        crop_path = os.path.join(root_dir, 'crop')
        if not os.path.exists(crop_path):
            os.makedirs(crop_path)
        else:
            os.system('rm -rf {}'.format(crop_path))
            os.makedirs(crop_path)
        image_path = os.path.join(video_path, 'images')
        point_path = os.path.join(video_path, 'annot')
        image_names = os.listdir(image_path)
        process_count = 0
        for idx2, image_name in enumerate(image_names):
            image_file = os.path.join(image_path, image_name)
            points_file = os.path.join(point_path, image_name.replace('jpg', 'pts'))
            point_lines = None
            with open(points_file, 'r') as f:
                point_lines = f.readlines()[3:71]
            image = cv2.imread(image_file)
            points = []
            for line in point_lines:
                x,y = line[:-1].split()
                x,y = int(float(x)), int(float(y))
                points.append([x,y])
            points = np.array(points)
            x_max = np.max(points[:,0])
            x_min = np.min(points[:,0])
            y_max = np.max(points[:,1])
            y_min = np.min(points[:,1])
            bbox = np.array([[x_min, y_min], [x_max, y_max]])
            bbox = adjust_bbox(bbox, (image.shape[0], image.shape[1]))
            
            if False:
                cv2.imshow('image', image)
                if cv2.waitKey(0) == ord('q'):
                    sys.exit(0)
            
            crop_image = image[bbox[0][1]:bbox[1][1], bbox[0][0]:bbox[1][0]].copy()
            crop_points = points.copy()
            crop_points[:,0] -= bbox[0][0]
            crop_points[:,1] -= bbox[0][1]
            if (crop_image.shape[0]<=0) or (crop_image.shape[1]<=0):
                print('[{}/{}] [{}/{}] crop image: {} {} {}'.format(
                    idx1, len(video_dir_names),
                    idx2, len(image_names),
                    crop_image.shape, video_dir_name, image_name
                ))
            
            for point in crop_points:
                points_str = str(point[0])
                points_str += '\t'
                points_str += str(point[1])
                points_str += '\t'
            points_str = points_str[0:-1]
            
            image_save_file = os.path.join(crop_path, image_name)
            cv2.imwrite(image_save_file, crop_image)
            points_save_file = os.path.join(crop_path, image.name.replace('jpg', 'pts'))
            with open(points_save_file, 'w+') as f:
                f.write(points_str)
                f.flush()
            process_count += 1
        print('[{}/{}] [{}/{}]'.format(
            idx1, len(video_dir_names),
            video_dir_name,
            process_count, len(image_names)
        ))

def resize_128(root_dir):
    video_dir_names = os.listdir(root_dir)
    for idx1, video_dir_name in enumerate(video_dir_names):
        video_path = os.path.join(root_dir, video_dir_name)
        resize_path = os.path.join(video_path, '128')
        if not os.path.exists(resize_path):
            os.makedirs(resize_path)
        else:
            os.system('rm -rf {}'.format(resize_path))
            os.makedirs(resize_path)
        crop_path = os.path.join(video_path, 'crop')
        crop_files = os.listdir(crop_path)
        image_list = []
        point_list = []
        for crop_file in crop_files:
            if crop_file.endswith('jpg'):
                image_list.append(os.path.join(crop_path, crop_file))
            else:
                point_list.append(os.path.join(crop_path, crop_file))
        process_count = 0
        for idx2, image_file in enumerate(image_list):
            points_file = image_file.replace('jpg', 'pts')
            point_lines = None
            with open(points_file, 'r') as f:
                point_lines = f.readlines()
            image = cv2.imread(image_file)
            points = np.fromstring(point_lines[0], dtype=np.int, sep='\t')
            
            image_128 = cv2.resize(image, (128, 128))
            scale = 128.0/image.shape[0]
            
            points = points.astype(np.float64)
            points *= scale
            points = points.astype(np.int)
            
            if points.shape[0] != 136:
                print('points shape not correct !!! {}'.format(image_file))
                if cv2.waitKey(0):
                    continue
            if (image_128.shape[0]<=0) or (image_128.shape[1]<=0):
                print('[{}/{}] [{}/{}] crop image: {} {} {}'.format(
                    idx1, len(video_dir_names),
                    idx2, len(image_list),
                    image_128.shape, video_dir_name, image_file
                ))
                if cv2.waitKey(1000):
                    continue
            
            if cv2.waitKey(1000) == ord('q'):
                sys.exit(0)
            
            points_str = ''
            for point in points:
                points_str = str(point)
                points_str += '\t'
            points_str = points_str[0:-1]
            
            image_save_file = image_file.replace('crop', '128')
            cv2.imwrite(image_save_file, image_128)
            points_save_file = image_save_file.replace('jpg', pts)
            with open(points_save_file, 'w+') as f:
                f.write(points_str)
                f.flush()
            process_count += 1
        print('[{}/{}] {} [{}/{}]'.format(
            idx1, len(video_dir_names),
            video_dir_name,
            process_count, len(image_list)
        ))


def crop_size(root_dir, shape=(128,128)):
    part_dir_names = os.listdir(root_dir)
    skip_list = []
    for idx1, part_dir_name in enumerate(part_dir_names):
        part_path = os.path.join(root_dir, part_dir_name)
        crop_path = os.path.join(root_dir, '{}_crop_{}_{}'.format(part_dir_name, shape[0], shape[1]))
        # temp_path = os.path.join(part_path, 'crop')
        # if os.path.exists(temp_path):
        #     os.system('rm -rf {}'.format(temp_path))
        # temp_path = os.path.join(part_path, '128')
        # if os.path.exists(temp_path):
        #     os.system('rm -rf {}'.format(temp_path))
        if not os.path.exists(crop_path):
            os.makedirs(crop_path)
        else:
            os.system('rm -rf {}'.format(crop_path))
            os.makedirs(crop_path)
        image_names = os.listdir(part_path)
        process_count = 0
        for idx2, image_name in enumerate(image_names):
            if not image_name.endswith('.png'):
                continue
            image_file = os.path.join(part_path, image_name)
            points_file = image_file.replace('png', 'pts')
            point_lines = None
            with open(points_file, 'r') as f:
                point_lines = f.readlines()[3:71]
            image = cv2.imread(image_file)
            points = []
            for line in point_lines:
                x,y = line[:-1].split()
                x,y = float(x), float(y)
                points.append([x,y])
            points = np.array(points)
            x_max = np.max(points[:,0])
            x_min = np.min(points[:,0])
            y_max = np.max(points[:,1])
            y_min = np.min(points[:,1])
            bbox = np.array([[x_min, y_min], [x_max, y_max]])
            bbox = adjust_bbox(bbox, (image.shape[0], image.shape[1]))
            
            if False:
                cv2.imshow('image', image)
                if cv2.waitKey(1000) == ord('q'):
                    sys.exit(0)
            
            crop_image = image[
                int(round(bbox[0][1])):int(round(bbox[1][1])),
                int(round(bbox[0][0])):int(round(bbox[1][0]))
            ].copy()
            crop_points = points.copy()
            crop_points[:,0] -= bbox[0][0]
            crop_points[:,1] -= bbox[0][1]
            if (crop_image.shape[0]<=0) or (crop_image.shape[1]<=0):
                print('[{}/{}] [{}/{}] crop image: {} {} {}'.format(
                    idx1, len(part_dir_names),
                    idx2, len(image_names),
                    crop_image.shape, part_dir_name, image_name
                ))
            
            if (crop_image is None) or (crop_image.shape[0]<=10) or (crop_image.shape[1] <= 10):
                skip_list.append(image_file)
                print('skip: ', image_file)
                continue
            image_ = cv2.resize(crop_image, shape)
            scales = (float(shape[0])/float(crop_image.shape[0]),
                      float(shape[1])/float(crop_image.shape[1]))
            points_ = crop_points*scales
            
            if True:
                show_image = image_.copy()
                for i in range(points_.shape[0]):
                    cv2.circle(show_image,(int(round(points_[i][0])),int(round(points_[i][1]))),1,(0, 255, 0),1)
                show_image = cv2.resize(show_image, (512, 512))
                cv2.imshow('img', show_image)
                if cv2.waitKey(300) == ord('q'):
                    sys.exit(0)
            
            points_str = ''
            for point in points_:
                points_str += str(point[0])
                points_str += '\t'
                points_str += str(point[1])
                points_str += '\t'
            points_str = points_str[0:-1]
            
            image_save_file = os.path.join(crop_path, image_name)
            cv2.imwrite(image_save_file, image_)
            points_save_file = image_save_file.replace('png', 'pts')
            with open(points_save_file, 'w+') as f:
                f.write(points_str)
                f.flush()
            process_count += 1
        print('[{}/{}] {} [{}/{}]'.format(
            idx1, len(part_dir_names),
            part_dir_name,
            process_count, len(image_names)
        ))

    print('skip list: ', skip_list)
    skip_str = ''
    for skip in skip_list:
        skip_str += skip
        skip_str += '\n'
    with open('./skip_list.txt', 'w+') as f:
        f.write(skip_str)
        f.flush()


def make_lst(root_dir, save_file):
    video_names = os.listdir(root_dir)
    cur_idx = 0
    infos = ''
    for idx, video_name in enumerate(video_names):
        print('[{}/{}] dealing with {}'.format(len(video_names), idx, video_name))
        video_path = os.path.join(root_dir, video_name)
        image_path = os.path.join(video_path, 'crop_128_128')
        files = os.listdir(image_path)
        for f in files:
            if f.endswith('jpg'):
                sub_path = video_name + '/crop_128_128/' + f
                points_file = os.path.join(image_path, f.replace('jpg', 'pts'))
                lines = []
                with open(points_file, 'r') as pf:
                    lines = pf.readlines()
                l = str(cur_idx) + '\t'
                l += lines[0] + '\t'
                l += sub_path
                l += '\n'
                infos += 1
                cur_idx += 1
    with open(save_file, 'w+') as sf:
        sf.write(infos)
        sf.flush()

def make_name_lst(root_dir, save_file):
    video_names = os.listdir(root_dir)
    cur_idx = 0
    infos = ''
    for idx, video_name in enumerate(video_names):
        print('[{}/{}] dealing with {}'.format(len(video_names), idx, video_name))
        video_path = os.path.join(root_dir, video_name)
        image_path = os.path.join(video_path, 'crop_128_128')
        files = os.listdir(image_path)
        for f in files:
            if f.endswith('jpg'):
                sub_path = video_name + '/crop_128_128/' + f
                infos += sub_path + '\n'
                cur_idx += 1
    with open(save_file, 'w+') as sf:
        sf.write(infos)
        sf.flush()


def show_list(image_list, point_list):
    for image_file in image_list:
        point_file = image_file.replace('jpg', 'pts')
        point_lines = None
        with open(point_file, 'r') as f:
            point_lines = f.readlines()
        points = np.fromstring(point_lines[0], dtype=np.int, sep='\t')
        pts = points.reshape((-1, 2))
        print('pts: ', pts.shape)
        image = cv2.imread(image_file)
        for p in pts:
            cv2.circle(image, (p[0], p[1]), 1, (0,0,255), 1)
        cv2.imshow('image', image)
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            sys.exit(0)

def show(root_dir):
    video_dir_names = os.listdir(root_dir)
    for video_dir_name in video_dir_names:
        video_path = os.path.join(root_dir, video_dir_name)
        image_path = os.path.join(video_path, 'images')
        point_path = os.path.join(video_path, 'annot')
        image_names = os.listdir(image_path)
        for image_name in image_names:
            image_file = os.path.join(image_path, image_name)
            points_file = os.path.join(point_path, image_name.replace('jpg', 'pts'))
            point_lines = None
            with open(points_file, 'r') as f:
                point_lines = f.readlines()
            image = cv2.imread(image_file)
            for line in point_lines:
                x,y = line[:-1].split()
                x,y = int(float(x)), int(float(y))
                print('x-y: ', x, y)
                cv2.circle(image, (x, y), 3, (0,0,255), 3)
            cv2.imshow('image', image)
            if cv2.waitKey(1000) & 0xFF == ord('q'):
                sys.exit(0)

def format_pts(point_list):
    # input format:
    # bbox_x1, bbox_y1, bbox_x2, bbox_y2, x1, y1, x2, y2, ..., x68, y68
    # output format:
    # x1, y1, x2, y2, ..., x68, y68
    for idx, point_file in enumerate(point_list):
        if not os.path.exists(point_file):
            continue
        line = None
        with open(point_file, 'r') as f:
            line = f.readlines()
        items = line[0].split('\t')
        items = items[4:]
        pts_str = ''
        for item in items:
            pts_str += item
            pts_str += '\t'
        pts_str = pts_str[0:-1]
        os.system('rm {}'.format(point_file))
        with open(point_file, 'w+') as f:
            f.write(pts_str)
            f.flush()


if __name__ == '__main__':
    
    root_dir = '/media/intellif/data/datasets/300w/300w'
    
    # step 1> crop and resize. crop_size(...) = crop(...) + resize_128(...)
    # crop(root_dir=root_dir)
    # resize_128(root_dir)
    crop_size(root_dir=root_dir, shape=(128, 128))
    
    # step 2> make im2rec.py lst
    make_lst(
        root_dir=root_dir,
        save_file=''
    )
    
    # show
    show(root_dir=root_dir)
    
    make_name_lst(
        root_dir=root_dir,
        save_file=''
    )