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
import argparse
import logging
import torch
import tqdm
import time
import shutil
import numpy as np
import pandas as pd

from sklearn.metrics import average_precision_score

from torchvision.transforms import ConvertImageDtype, Normalize, Resize, Compose
from torchvision.transforms.functional import InterpolationMode
from torchvision.io import ImageReadMode, read_image

from swiss.dataset_builders.icubworld import ICWDatasetBuilder, generate_icb_episodes
from swiss.dataset_builders.perseg import PerSegDatasetBuilder, generate_perseg_episodes

from swiss.kmeans_utils import kmeans_map_from_feature_map_full
from swiss.feature_extractor_utils import generate_features_dino
from swiss.engine import SwissEngine

# Schema for the few-shot detection dataset
IMAGE_PATH_COLUMN_NAME = "image_path"
LABEL_COLUMN_NAME = "labels"

IMAGE_RESIZE = 448
RANDOM_SEED = 42
PERCENTILE_FOR_ADAPTIVE_THRESHOLD = 5
COORDINATE_SCALING_FACTOR = 200
KMEANS_CLUSTER_NUM_QUERY = 150


def process_dataset(
        dataset, fs_episode, dataset_classes, draw_masks=False,
        feature_extractor_type="vit_b", output_masks_dir="output_dir",
        annotation_type='bbox', refine_patch_maps=False, verbose=False):

    device = torch.device("cuda")

    # Initializing DINOv2 model
    image_transforms = Compose([
        Resize([IMAGE_RESIZE, IMAGE_RESIZE], interpolation=InterpolationMode.BICUBIC, antialias=True),
        ConvertImageDtype(torch.float),
        Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]
    )
    if feature_extractor_type == "vit_s":
        feature_extractor_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    elif feature_extractor_type == "vit_b":
        feature_extractor_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
    elif feature_extractor_type == "vit_l":
        feature_extractor_model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    feature_extractor_model.to(device)
    feature_extractor_model.eval()
    feature_map_generator = (lambda images: generate_features_dino(
        images, feature_extractor_model, image_transforms, device=device))

    swiss_engine = SwissEngine(
        dataset_classes=dataset_classes, annotation_type=annotation_type,
        refine_patch_maps = refine_patch_maps,
        percentile_for_adaptive_threshold=PERCENTILE_FOR_ADAPTIVE_THRESHOLD)

    if draw_masks:
        output_dir_base = os.path.join(os.getcwd(), output_masks_dir)
        output_dir_supp = os.path.join(output_dir_base, "AA_support_dir")
        if os.path.isdir(output_dir_base):
            try:
                shutil.rmtree(output_dir_base)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        os.mkdir(output_dir_base)
        os.mkdir(output_dir_supp)

    df_columns = ["supp_class_name", "swiss_iou", "swiss_ap", "swiss_acc"]

    results_df = pd.DataFrame(columns=df_columns)
    results_df = results_df.set_index('supp_class_name')

    logging.info(f"Starting evaluation")

    support_rows = dataset.select(fs_episode['support'])
    supp_labels = support_rows[LABEL_COLUMN_NAME]
    supp_img_paths = support_rows[IMAGE_PATH_COLUMN_NAME]
    supp_images = [read_image(img_path, mode=ImageReadMode.RGB) for img_path in supp_img_paths]
    supp_feature_maps, _ = feature_map_generator(supp_images)

    query_rows = dataset.select(fs_episode['query'])
    query_labels = query_rows[LABEL_COLUMN_NAME]
    query_img_paths = query_rows[IMAGE_PATH_COLUMN_NAME]
    query_images = [read_image(img_path, mode=ImageReadMode.RGB) for img_path in query_img_paths]

    if annotation_type == 'bbox':
        BOUNDING_BOX_COLUMN_NAME = "bounding_box"
        supp_bboxes = support_rows[BOUNDING_BOX_COLUMN_NAME]
        query_bboxes = query_rows[BOUNDING_BOX_COLUMN_NAME]
    elif annotation_type == 'segmap':
        SEGMENTATION_PATH_COLUMN_NAME = "segmentation_path"
        supp_seg_paths = support_rows[SEGMENTATION_PATH_COLUMN_NAME]
        query_seg_paths = query_rows[SEGMENTATION_PATH_COLUMN_NAME]
    else:
        raise ValueError('Unknown annotation type')

    # Query feature generation
    generate_start_time = time.time()
    query_feature_maps, _ = feature_map_generator(query_images)
    if verbose:
        logging.info(f"Finished generating query features, avg time: {(time.time()-generate_start_time)/len(query_rows)}")

    # Kmeans clustering
    if refine_patch_maps:
        kmeans_start_time = time.time()
        query_kmeans_maps = []
        for qfm in tqdm.tqdm(query_feature_maps, desc=f"Query clustering"):
            query_kmeans_maps.append(
                kmeans_map_from_feature_map_full(
                    qfm, cluster_num=KMEANS_CLUSTER_NUM_QUERY, coord_scaling_factor=COORDINATE_SCALING_FACTOR))
        logging.info(f"Finished kmeans for queries, avg time: {(time.time()-kmeans_start_time)/len(query_rows)}")
        logging.info("")

    gt_logits = np.zeros((len(support_rows), len(query_rows)))
    swiss_iou = np.zeros((len(support_rows), len(query_rows)))
    swiss_logits = np.zeros((len(support_rows), len(query_rows)))
    s_q_pairs = [(s_ind, q_ind) for s_ind in range(len(support_rows)) for q_ind in range(len(query_rows))]

    supp_start_time = time.time()

    for s_row_ind, q_row_ind in tqdm.tqdm(s_q_pairs, desc=f"Processing support/query pairs."):

        class_name = dataset_classes[supp_labels[s_row_ind]]

        populate_support_kwargs = {
            'image': supp_images[s_row_ind], 'label': supp_labels[s_row_ind],
            'feature_map': supp_feature_maps[s_row_ind]}
        if annotation_type=='bbox':
            populate_support_kwargs['bounding_box'] = supp_bboxes[s_row_ind]
        elif annotation_type == 'segmap':
            populate_support_kwargs['segmentation_map_path'] = supp_seg_paths[s_row_ind]
        swiss_engine.populate_support_sample(**populate_support_kwargs)
        if draw_masks:
            swiss_engine.draw_support_mask(draw_path=output_dir_supp, support_image_path=supp_img_paths[s_row_ind])
            output_dir_ep_supp = os.path.join(output_dir_base, f'supp_{class_name}')
            os.mkdir(output_dir_ep_supp)

        predict_query_kwargs = {'feature_map': query_feature_maps[q_row_ind]}
        if refine_patch_maps:
            predict_query_kwargs['kmeans_map'] = query_kmeans_maps[q_row_ind]
        pred_logit, pred_seg_mask = swiss_engine.predict_mask_for_query_sample(**predict_query_kwargs)

        is_true_positive_sample = (supp_labels[s_row_ind] == query_labels[q_row_ind])
        if draw_masks and is_true_positive_sample:
            swiss_engine.draw_predicted_query_mask(
                draw_path=output_dir_ep_supp, query_image_path=query_img_paths[q_row_ind],
                query_label=query_labels[q_row_ind])

        # Updating metrics
        gt_logits[s_row_ind, q_row_ind] = is_true_positive_sample
        swiss_logits[s_row_ind, q_row_ind] = pred_logit
        if annotation_type == 'bbox':
            swiss_iou[s_row_ind, q_row_ind] = swiss_engine.evaluate_iou_bbox(
                pred_seg_mask, query_bboxes[q_row_ind], query_images[q_row_ind])
        elif annotation_type == 'segmap':
            swiss_iou[s_row_ind, q_row_ind] = swiss_engine.evaluate_iou_segmap(
                pred_seg_mask, query_seg_paths[q_row_ind])

    if verbose:
        logging.info(f"Finished support/query comparison, avg time: {(time.time()-supp_start_time)/len(s_q_pairs)}")
        logging.info("")

    # Computing metrics
    swiss_pred_labels = np.argmax(swiss_logits, axis=0)

    for s_ind in range(len(support_rows)):
        s_class_name = dataset_classes[supp_labels[s_ind]]
        gt_mask = gt_logits[s_ind].astype(bool)

        swiss_avg_iou = np.mean(swiss_iou[s_ind][gt_mask])
        swiss_acc = np.mean(swiss_pred_labels[gt_mask]==s_ind)
        swiss_ap = average_precision_score(gt_mask, swiss_logits[s_ind])

        results_df.loc[s_class_name] = [swiss_avg_iou, swiss_ap, swiss_acc]

        if verbose:
            logging.info(f"Class: {s_class_name}, SwissIOU: {swiss_avg_iou}, ")
            logging.info(f"SwissAP: {swiss_ap}, SwissAcc: {swiss_acc}")
            logging.info("")

    logging.info(f"Finished evaluation")
    logging.info(f"Current average SwissIOU is {np.mean(results_df['swiss_iou'])}")
    logging.info(f"Current average SwissAP is {np.mean(results_df['swiss_ap'])}")
    logging.info(f"Current average SwissACC is {np.mean(results_df['swiss_acc'])}")

    return results_df


def main(args):

    with open(args.log_file, 'w') as f:
        f.close()
    logging.basicConfig(
        filename=args.log_file, filemode='a', format='%(message)s',
        datefmt='%H:%M:%S', level=logging.INFO)

    logging.info(
        f"Starting SWISS DINO comparison on {args.dataset_name} dataset. "
        f"Feature extractor DINOv2_{args.fe_model_type}")
    logging.info(
        f"Hyperparameters: PERCENTILE_FOR_ADAPTIVE_THRESHOLD {PERCENTILE_FOR_ADAPTIVE_THRESHOLD}")
    if args.refine_patch_maps:
        logging.info(
            f"Refinement hyperparameters: KMEANS_CLUSTER_NUM_QUERY {KMEANS_CLUSTER_NUM_QUERY}, "
            f"COORDINATE_SCALING_FACTOR {COORDINATE_SCALING_FACTOR} ")
    logging.info("")

    if args.dataset_name == 'icubworld':
        dataset_builder = ICWDatasetBuilder(data_dir=args.data_dir)
        fs_dataset = dataset_builder.as_dataset(split="validation")
        fs_episodes = generate_icb_episodes(
            fs_dataset, episode_num=1, n_way=None,
            seed=RANDOM_SEED, cluttered_scenes=args.cluttered_scenes)
        annotation_type = 'bbox'
    elif args.dataset_name == 'perseg':
        dataset_builder = PerSegDatasetBuilder(data_dir=args.data_dir)
        fs_dataset = dataset_builder.as_dataset(split="validation")
        fs_episodes = generate_perseg_episodes(
            fs_dataset, episode_num=1, n_way=None, seed=RANDOM_SEED)
        annotation_type = 'segmap'

    dataset_classes = dataset_builder.classes
    # Assume one episode for evaluation
    fs_episode = fs_episodes[0]
    process_dataset(
        fs_dataset, fs_episode, dataset_classes,
        draw_masks=args.draw_masks,
        feature_extractor_type=args.fe_model_type,
        output_masks_dir=args.output_masks_dir,
        annotation_type=annotation_type,
        refine_patch_maps=args.refine_patch_maps,
        verbose=args.verbose)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SwissDINO on datasets.')
    parser.add_argument('--dataset_name', type=str) # options: perseg, icubworld
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--fe_model_type', type=str) # options: vit_s, vit_b, vit_l
    parser.add_argument('--draw_masks', action='store_true')
    parser.set_defaults(draw_masks=False)
    parser.add_argument('--output_masks_dir', type=str, default="output_dir")
    parser.add_argument('--log_file', type=str, default="log.txt")
    parser.add_argument('--cluttered_scenes', action='store_true') # for icubworld dataset
    parser.set_defaults(cluttered_scenes=False)
    parser.add_argument('--refine_patch_maps', action='store_true')
    parser.set_defaults(refine_patch_maps=False)
    parser.add_argument('--verbose', action='store_true')
    parser.set_defaults(verbose=False)
    args = parser.parse_args()
    main(args)
