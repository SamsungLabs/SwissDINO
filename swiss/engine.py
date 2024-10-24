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

import os
import cv2
import numpy as np
from PIL import Image

from .kmeans_utils import kmeans_map_from_feature_map_with_bbox
from .segmentation_utils import get_threshold, semseg_logit_map, refine_semseg_with_kmeans
from .draw_utils import draw_patch_mask
from .feature_extractor_utils import compare_feature_maps
from .bbox_utils import bbox_iou, seg_map_to_bbox, seg_mask_iou



class SwissEngine(object):

    def __init__(
            self, dataset_classes, annotation_type, refine_patch_maps=False,
            percentile_for_adaptive_threshold=5,
            segmentation_mask_binarization_threshold=0.1):

        self.dataset_classes = dataset_classes
        self.annotation_type = annotation_type
        self.refine_patch_maps = refine_patch_maps
        self.percentile_for_adaptive_threshold = percentile_for_adaptive_threshold
        self.segmentation_mask_binarization_threshold = segmentation_mask_binarization_threshold

        self.supp_class_name = None
        self.supp_feature_map = None
        self.patch_num = None
        self.support_prototype = None
        self.threshold = None
        self.pred_logit = None
        self.pred_seg_mask = None


    def populate_support_sample(
            self, image, label, feature_map, bounding_box=None, segmentation_map_path=None):

        self.supp_class_name = self.dataset_classes[label]
        self.supp_feature_map = feature_map
        self.patch_num = feature_map.shape[0]

        if self.annotation_type == 'bbox':
            assert bounding_box is not None
            img_h, img_w = image.shape[1:]
            x_min, y_min, x_max, y_max = bounding_box
            supp_bbox_xyxy_patch = [
                (x_min*self.patch_num)//img_w, (y_min*self.patch_num)//img_h,
                (x_max*self.patch_num)//img_w + 1, (y_max*self.patch_num)//img_h + 1]
            # Refine support bbox with kmeans
            supp_seg_patch_map = kmeans_map_from_feature_map_with_bbox(feature_map, supp_bbox_xyxy_patch)
            if np.sum(supp_seg_patch_map) < 1:
                raise ValueError(f"Support image for class {self.supp_class_name} could not be segmented for objects")

        elif self.annotation_type == 'segmap':
            assert segmentation_map_path is not None
            supp_seg_arrray = np.array(Image.open(segmentation_map_path).convert('L'))
            supp_seg_resized_patch = cv2.resize(supp_seg_arrray, (self.patch_num, self.patch_num), interpolation = cv2.INTER_AREA)
            supp_seg_patch_map = (supp_seg_resized_patch > self.segmentation_mask_binarization_threshold)

        feat_dim = feature_map.shape[2]
        support_prototype = np.mean(feature_map[supp_seg_patch_map].reshape(-1, feat_dim), axis=0)
        self.support_prototype = support_prototype / np.linalg.norm(support_prototype)

        supp_self_logits = semseg_logit_map(feature_map, self.support_prototype)
        self.threshold = get_threshold(
            supp_self_logits, supp_seg_patch_map, adaptive_threshold_percentile=self.percentile_for_adaptive_threshold)


    def draw_support_mask(self, draw_dir, support_image_path):
        img_resize = 448
        supp_self_logits = semseg_logit_map(self.supp_feature_map, self.support_prototype)
        supp_mask_patch = (supp_self_logits > self.threshold)

        supp_img_cv2 = cv2.imread(support_image_path)
        supp_img_resized = cv2.resize(supp_img_cv2, (img_resize, img_resize), interpolation=cv2.INTER_CUBIC)
        support_img_with_coords = draw_patch_mask(supp_img_resized, supp_mask_patch, image_resize=img_resize)

        supp_base_name = support_image_path.split(os.sep)[-1].split('.')[0]
        cv2.imwrite(
            os.path.join(draw_dir, f'AA_{self.supp_class_name}_{supp_base_name}_support_mask.jpg'),
            support_img_with_coords)


    def predict_mask_for_query_sample(self, feature_map, kmeans_map=None):

        query_seg_mask_logits = semseg_logit_map(feature_map, support_prototype=self.support_prototype)
        pred_seg_mask_patch = (query_seg_mask_logits >= self.threshold)
        persam_mask = (query_seg_mask_logits == np.max(query_seg_mask_logits))
        if np.sum(pred_seg_mask_patch) == 0:
            # PerSAM approach:
            pred_seg_mask_patch = persam_mask

        # Split the seg map into connected components and choose the best matching component.
        num_components, conn_comp_labels, _, _ = cv2.connectedComponentsWithStats(
            pred_seg_mask_patch.astype(np.uint8), connectivity=8)
        candidate_masks = []
        mask_class_scores = []
        for comp_ind in range(1, num_components):
            seg_mask = np.where(conn_comp_labels == comp_ind, True, False)
            if self.refine_patch_maps:
                seg_mask = refine_semseg_with_kmeans(kmeans_map, seg_mask)
            mask_class_score = compare_feature_maps(feature_map, seg_mask, self.support_prototype)
            mask_class_scores.append(mask_class_score)
            candidate_masks.append(seg_mask)

        best_class_mask_ind = np.argmax(mask_class_scores)
        self.pred_logit = mask_class_scores[best_class_mask_ind]
        self.pred_seg_mask = candidate_masks[best_class_mask_ind]
        return self.pred_logit, self.pred_seg_mask


    def draw_predicted_query_mask(self, draw_dir, query_image_path, query_label):
        img_resize = 448
        query_base_name = f"{self.dataset_classes[query_label]}_{query_image_path.split(os.sep)[-2]}"
        query_base_name = f"{query_base_name}_{query_image_path.split(os.sep)[-1].split('.')[0]}"

        query_img = cv2.imread(query_image_path)
        query_img_resized = cv2.resize(query_img, (img_resize, img_resize), interpolation=cv2.INTER_CUBIC)
        query_img_with_coords = draw_patch_mask(query_img_resized, self.pred_seg_mask)
        cv2.imwrite(os.path.join(draw_dir, f'{query_base_name}.jpg'), query_img_with_coords)


    def evaluate_iou_bbox(self, pred_seg_mask, query_bbox, query_img):
        pred_bbox = seg_map_to_bbox(pred_seg_mask)
        img_h, img_w = query_img.shape[1:]
        x_min, y_min, x_max, y_max = query_bbox
        query_gt_bb_resized = [
            (x_min*self.patch_num)//img_w, (y_min*self.patch_num)//img_h,
            (x_max*self.patch_num)//img_w + 1, (y_max*self.patch_num)//img_h + 1]
        return bbox_iou(pred_bbox, query_gt_bb_resized)


    def evaluate_iou_segmap(self, pred_seg_mask, query_segmap_path):
        query_gt_seg_array = np.array(Image.open(query_segmap_path).convert('L'))
        query_gt_seg_mask_patch = cv2.resize(query_gt_seg_array, (self.patch_num, self.patch_num), interpolation = cv2.INTER_AREA)
        query_gt_seg_mask_patch = (query_gt_seg_mask_patch > self.segmentation_mask_binarization_threshold)
        return seg_mask_iou(pred_seg_mask, query_gt_seg_mask_patch)