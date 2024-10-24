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


def seg_map_to_bbox(seg_map):
    h, w = seg_map.shape
    row_nonzero_inds = np.arange(h)[np.where(np.sum(seg_map, axis=1)>0, True, False)]
    col_nonzero_inds = np.arange(w)[np.where(np.sum(seg_map, axis=0)>0, True, False)]
    bbox_xyxy = [col_nonzero_inds[0], row_nonzero_inds[0], col_nonzero_inds[-1]+1, row_nonzero_inds[-1]+1]
    return bbox_xyxy


def bbox_iou(left_bbox_xyxy, right_bbox_xyxy):
    int_x_min = max(left_bbox_xyxy[0], right_bbox_xyxy[0])
    int_x_max = min(left_bbox_xyxy[2], right_bbox_xyxy[2])
    int_w = max(0, int_x_max-int_x_min)
    int_y_min = max(left_bbox_xyxy[1], right_bbox_xyxy[1])
    int_y_max = min(left_bbox_xyxy[3], right_bbox_xyxy[3])
    int_h = max(0, int_y_max-int_y_min)
    int_area = int_w * int_h
    left_area = (left_bbox_xyxy[2]-left_bbox_xyxy[0]) * (left_bbox_xyxy[3]-left_bbox_xyxy[1])
    right_area = (right_bbox_xyxy[2]-right_bbox_xyxy[0]) * (right_bbox_xyxy[3]-right_bbox_xyxy[1])
    union_area = left_area + right_area -int_area
    return int_area / union_area

def seg_mask_iou(left_seg_mask, right_seg_mask):
   int_area = np.sum(left_seg_mask*right_seg_mask)
   union_area = np.sum(left_seg_mask) + np.sum(right_seg_mask) - int_area
   return int_area / union_area