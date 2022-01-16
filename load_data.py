from torch.utils import data
import os
import re
import os.path
import cv2
import numpy as np
import random
import torch
from model.base_component import DataAugmentationForAttInMVS


def generate_data_path(data_folder_path: str, state='train', multi_views=3):
    """
        Generate datafile path where the samples are arranged in this way:
        cameras:
            data_folder_path/Cameras/00000000_cam.txt
            data_folder_path/Cameras/00000001_cam.txt
            ...
            data_folder_path/Cameras/000000063_cam.txt
            data_folder_path/Cameras/pair.txt
        images:
            data_folder_path/Rectified/scan1_train/rect_001_0_r5000.png ~ data_folder_path/Rectified/scan1_train/rect_001_6_r5000.png
            data_folder_path/Rectified/scan1_train/rect_002_0_r5000.png ~ data_folder_path/Rectified/scan1_train/rect_002_6_r5000.png
            ...
            data_folder_path/Rectified/scan1_train/rect_049_0_r5000.png ~ data_folder_path/Rectified/scan1_train/rect_049_6_r5000.png
            ...
            data_folder_path/Rectified/scan128_train/rect_001_0_r5000.png ~ data_folder_path/Rectified/scan128_train/rect_001_6_r5000.png
            data_folder_path/Rectified/scan128_train/rect_002_0_r5000.png ~ data_folder_path/Rectified/scan128_train/rect_002_6_r5000.png
            ...
            data_folder_path/Rectified/scan128_train/rect_049_0_r5000.png ~ data_folder_path/Rectified/scan128_train/rect_049_6_r5000.png
        depth map:
            data_folder_path/Depths/scan1_train/depth_map_0000.pfm ~ data_folder_path/Depths/scan1_train/depth_map_0048.pfm
            ...
            data_folder_path/Depths/scan128_train/depth_map_0000.pfm ~ data_folder_path/Depths/scan128_train/depth_map_0048.pfm

        Based on the assumption that the information of geometric transformation has been included in the correspondence of
        a reference image and several source images, we extract the information of geometric transformation
        using attention mechanism rather than projection using cameras' parameters.
        So we wouldn't load cameras' parameters exclude 'pair.txt' which contains the correspondence of ref images and source images

        There have 1 reference image and 2 source images in each image group
        In training dataset,
        there have 79 scans, each scans have 49 different views, each view have 7 images in different light confidence,
        it means we all have 79 x 49 x 7 = 27097 images in training dataset.
        For training, we choose 3 images from different light confidence of same view of same scan,
        so we all have 79 x 49 x 7 = 27097 image groups for training, which contains 1 reference image and 2 source images.
        The correspondences of reference and source images are defined in pair.txt

        :param data_folder_path: data folder path
        :param state: mode, 'train', 'validate' or 'test'
        :param multi_views: the number of multi views
        :return: list of [[ref_path, source_1_path, ..., source_n-1_path, depth_path]]

    """
    sample_list = []
    pair_file_path = data_folder_path + '/Cameras/pair.txt'
    pair_list = open(pair_file_path).read().split()
    # train_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
    #              45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
    #              74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
    #              101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
    #              121, 122, 123, 124, 125, 126, 127, 128]
    train_set = [2]
    # validate_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]
    validate_set = [3]
    test_set = [117]
    if state == 'train':
        data_set = train_set
    elif state == 'validate':
        data_set = validate_set
    else:
        data_set = test_set

    # Generate data file path for each scan
    for i in data_set:
        image_folder = os.path.join(data_folder_path, ('Rectified/scan%d_train' % i)).replace('\\', '/')
        depth_folder = os.path.join(data_folder_path, ('Depths/scan%d_train' % i)).replace('\\', '/')

        # for each light confidence
        for j in range(0, 7):
            # for each reference image
            for p in range(int(pair_list[0])):
                paths = []
                # Get reference image using pair.txt
                ref_index = int(pair_list[22 * p + 1])
                ref_image_path = os.path.join(image_folder, 'rect_%03d_%d_r5000.png' % ((ref_index + 1), j)).replace(
                    '\\', '/')
                paths.append(ref_image_path)
                # Get source images using pair.txt
                for view in range(multi_views - 1):
                    view_index = int(pair_list[22 * p + view * 2 + 3])
                    view_image_path = os.path.join(image_folder,
                                                   'rect_%03d_%d_r5000.png' % ((view_index + 1), j)).replace('\\', '/')
                    paths.append(view_image_path)
                # Get depth map path
                depth_image_path = os.path.join(depth_folder, 'depth_map_%04d.pfm' % ref_index).replace('\\', '/')
                paths.append(depth_image_path)
                sample_list.append(paths)
    return sample_list


def dataset_path_iter(dataset_path_list, batch_size):
    num_examples = len(dataset_path_list)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = indices[i: min(i + batch_size, num_examples)]
        # batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
        yield i, list(dataset_path_list[index] for index in batch_indices)


def load_pfm(pfm_file):
    """ Load depth map """
    header = pfm_file.readline().decode('UTF-8').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(pfm_file.readline().decode('UTF-8').rstrip())
    if scale < 0:
        data_type = '<f'
    else:
        data_type = '>f'
    data_str = pfm_file.read()
    data_content = np.fromstring(data_str, data_type)
    shape = (height, width, 3) if color else (height, width)
    data_content = np.reshape(data_content, shape)
    data_content = cv2.flip(data_content, 0)
    return data_content


def load_dataset(data_path, multi_views):
    transform = DataAugmentationForAttInMVS()
    images_list = []
    depth_maps = []
    size = len(data_path)
    for i in range(size):
        images = []
        for view in range(multi_views):
            image = cv2.imread(data_path[i][view])
            image = transform(image)
            images.append(image)
        depth_map = load_pfm(open(data_path[i][multi_views], 'rb'))

        images = np.stack(images, axis=0)
        images_list.append(images)
        depth_maps.append(depth_map)
    return torch.tensor(images_list, dtype=torch.float32), torch.tensor(depth_maps, dtype=torch.float32)
