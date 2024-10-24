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


def semseg_logit_map(query_feature_map, support_prototype):
    query_fm_normed = query_feature_map / np.linalg.norm(query_feature_map, axis=-1, keepdims=True)
    return query_fm_normed @ support_prototype.T


def get_threshold(logit_map, segmentation_mask, adaptive_threshold_percentile=5, filter_out_background=True):
    # threshold = np.percentile(supp_self_logits[supp_gt_seg_mask_patch].flatten(), PERCENTILE_FOR_ADAPTIVE_THRESHOLD)
    background_patches_max_logit = np.percentile(
        logit_map[np.logical_not(segmentation_mask)].flatten(), 100-adaptive_threshold_percentile)
    object_patches_min_logit = np.percentile(
        logit_map[segmentation_mask].flatten(), adaptive_threshold_percentile)
    if filter_out_background:
        return max(background_patches_max_logit, object_patches_min_logit)
    return object_patches_min_logit


def refine_semseg_with_kmeans(kmeans_labels, pred_seg_mask):
    # Choose a set of connected components
    patch_num = kmeans_labels.shape[0]
    positive_clusters = np.unique(kmeans_labels[pred_seg_mask])
    ref_seg_mask = np.zeros((patch_num, patch_num), dtype=bool)
    for pos_cl_ind in positive_clusters:
        ref_seg_mask = np.logical_or(ref_seg_mask, (kmeans_labels == pos_cl_ind))

    return ref_seg_mask
