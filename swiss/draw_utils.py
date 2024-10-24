################################################################################
# Copyright (c) 2024 Samsung Electronics Co., Ltd.
#
# Author(s):
# Kirill Paramonov (k.paramonov@samsung.com)
# Jia-Xing Zhong (jiaxing.zhong@cs.ox.ac.uk)
# Umberto Michieli (u.michieli@samsung.com)
# Jijoong Moon (jijoong.moon@samsung.com)
# Mete Ozay (m.ozay@samsung.com)
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) License, (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at https://creativecommons.org/licenses/by-nc-sa/4.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
# For conditions of distribution and use, see the accompanying LICENSE.md file.
################################################################################

import numpy as np
import cv2


def draw_patch_mask(image, patch_mask, image_resize=448):
    patch_num = patch_mask.shape[0]
    coordinate_array = np.array([[[i,j] for j in range(patch_num)] for i in range(patch_num)])
    match_patch_inds = coordinate_array[patch_mask].reshape(-1, 2)
    patch_size = image_resize // patch_num
    resized_patch_inds = match_patch_inds * patch_size + patch_size//2
    start_color = np.array([255, 0, 0])
    end_color = np.array([0, 0, 255])

    for patch_ind, (patch_i, patch_j) in enumerate(resized_patch_inds):
        x,y = patch_j, patch_i
        point_color = start_color + (end_color - start_color) * patch_ind / len(resized_patch_inds)
        point_color = (int(point_color[0]), int(point_color[1]), int(point_color[2]))
        image = cv2.circle(image, (x,y), radius=3, color=point_color, thickness=-1)

    return image


def draw_bbox(image, bbox_xyxy):
    x_min, y_min, x_max, y_max = bbox_xyxy
    rec_color = (255, 0, 255)
    rec_thickness = 1
    return cv2.rectangle(image, (x_min, y_min), (x_max, y_max), rec_color, rec_thickness)
