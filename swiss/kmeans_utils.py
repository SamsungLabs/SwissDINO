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

from sklearn.cluster import KMeans
import numpy as np


def kmeans_map_from_feature_map_full(feature_map, cluster_num=100, coord_scaling_factor=100):

    fm_h, fm_w = feature_map.shape[:2]

    coordinate_array = np.array(
        [[[float(x)/fm_w, float(y)/fm_h] for x in range(fm_w)] for y in range(fm_h)])
    f_map_coord_augmented = np.dstack(
        (feature_map, coordinate_array * coord_scaling_factor))
    f_map_flat = f_map_coord_augmented.reshape(fm_h*fm_w, -1)

    clustering_alg = KMeans(n_clusters=cluster_num, random_state=0)
    clustering_alg.fit(f_map_flat)
    return clustering_alg.labels_.reshape(fm_h, fm_w)


def kmeans_map_from_feature_map_with_bbox(feature_map, bbox_xyxy_patch, cluster_num=5):

    x_min, y_min, x_max, y_max = bbox_xyxy_patch
    fm_h, fm_w, feat_dim = feature_map.shape

    boxed_patches = feature_map[y_min:y_max, x_min:x_max].reshape(-1, feat_dim)
    border_patches = np.vstack((
        feature_map[y_min-1, x_min:x_max], feature_map[y_max, x_min:x_max],
        feature_map[y_min:y_max, x_min-1], feature_map[y_min:y_max, x_max])).reshape(-1, feat_dim)

    kmeans_alg = KMeans(n_clusters=cluster_num, random_state=0)
    kmeans_alg.fit(boxed_patches)
    cluster_inds_array = kmeans_alg.labels_.reshape(y_max-y_min, x_max-x_min)
    negative_labels = kmeans_alg.predict(border_patches)
    negative_clusters = np.unique(negative_labels)

    ref_seg_mask = np.zeros((y_max-y_min, x_max-x_min), dtype=bool)
    for cl_ind in range(cluster_num):
        if cl_ind in negative_clusters:
            continue
        ref_seg_mask = np.logical_or(ref_seg_mask, cluster_inds_array==cl_ind)

    patch_mask = np.zeros((fm_h, fm_w), dtype=bool)
    patch_mask[y_min:y_max, x_min:x_max] = ref_seg_mask

    return patch_mask