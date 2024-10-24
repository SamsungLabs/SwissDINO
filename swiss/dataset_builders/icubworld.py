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
import datasets
import xml.etree.ElementTree as ET
import random


# Schema for the few-shot detection dataset
IMAGE_PATH_COLUMN_NAME = "image_path"
LABEL_COLUMN_NAME = "labels"
COARSE_LABEL_COLUMN_NAME = "coarse_labels"
BOUNDING_BOX_COLUMN_NAME = "bounding_box"
SUPPORT_BOOL_COLUMN_NAME = "is_support_image"
CLUTTERED_BOOL_COLUMN_NAME = "is_cluttered_image"


# Name of the dataset usually match the script name with CamelCase instead of snake_case
class ICWDatasetBuilder(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig

    def __init__(self, *args, **kwargs):

        self.dataset_name = "ICubWorld"
        if 'config_name' not in kwargs.keys():
            kwargs['config_name'] = 'default'
        dataset_cache_dir = os.path.join(os.getcwd(), "datasets_cache")
        kwargs['cache_dir'] = dataset_cache_dir

        super().__init__(*args, **kwargs)

        self.image_dir = os.path.join(self.config.data_dir, "Images")
        self.annotation_dir = os.path.join(self.config.data_dir, "Annotations_manual")
        self.cluttered_image_dir = os.path.join(self.config.data_dir, "Sequences_images")
        self.cluttered_annotation_dir = os.path.join(self.config.data_dir, "Sequences_annotations")
        self.coarse_classes = os.listdir(self.annotation_dir)
        self.classes = []
        for c_class in self.coarse_classes:
            f_classes = os.listdir(os.path.join(self.annotation_dir, c_class))
            self.classes.extend(f_classes)

        self.download_and_prepare()


    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset

        feature_dict = {
            IMAGE_PATH_COLUMN_NAME: datasets.Value("string"),
            LABEL_COLUMN_NAME: datasets.Value("int32"),
            COARSE_LABEL_COLUMN_NAME: datasets.Value("int32"),
            BOUNDING_BOX_COLUMN_NAME: [datasets.Value(dtype='int32')],
            SUPPORT_BOOL_COLUMN_NAME: datasets.Value("bool"),
            CLUTTERED_BOOL_COLUMN_NAME: datasets.Value("bool")
        }

        features = datasets.Features(feature_dict)
        return datasets.DatasetInfo(features=features)


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        dataset_val = datasets.SplitGenerator(name=datasets.Split.VALIDATION)

        return [dataset_val]


    def _generate_examples(self, **kwargs):

        ex_id = 0
        for c_class in self.coarse_classes:
            fine_classes = os.listdir(os.path.join(self.annotation_dir, c_class))
            for f_class in fine_classes:

                ann_path = os.path.join(self.annotation_dir, c_class, f_class)
                mix_prefix = os.listdir(ann_path)[0]
                ann_path = os.path.join(ann_path, mix_prefix)
                day_prefix = os.listdir(ann_path)[0]
                ann_path = os.path.join(ann_path, day_prefix)
                left_prefix = os.listdir(ann_path)[0]
                ann_path = os.path.join(ann_path, left_prefix)
                is_support = True
                for ann_file_name in sorted(os.listdir(ann_path)):

                    if ann_file_name[0] == '.':
                        continue
                    base_file_name = ann_file_name.split('.')[0]
                    ann_tree = ET.parse(os.path.join(ann_path, ann_file_name))
                    bbox_elt = ann_tree.getroot().find('object').find('bndbox')
                    bbox_xyxy_tags = ['xmin', 'ymin', 'xmax', 'ymax']
                    bbox_xyxy = [int(bbox_elt.find(tag).text) for tag in bbox_xyxy_tags]
                    image_path = os.path.join(
                        self.image_dir, c_class, f_class, mix_prefix, day_prefix, left_prefix, f'{base_file_name}.jpg')
                    example = {
                        IMAGE_PATH_COLUMN_NAME: image_path,
                        LABEL_COLUMN_NAME: self.classes.index(f_class),
                        COARSE_LABEL_COLUMN_NAME: self.coarse_classes.index(c_class),
                        BOUNDING_BOX_COLUMN_NAME: bbox_xyxy,
                        SUPPORT_BOOL_COLUMN_NAME: is_support,
                        CLUTTERED_BOOL_COLUMN_NAME: False
                    }
                    yield (ex_id, example)
                    is_support = False
                    ex_id += 1

        for scene_prefix in os.listdir(self.cluttered_annotation_dir):

            for ann_file_name in os.listdir(os.path.join(self.cluttered_annotation_dir, scene_prefix)):

                base_file_name = ann_file_name.split('.')[0]
                ann_tree = ET.parse(os.path.join(self.cluttered_annotation_dir, scene_prefix, ann_file_name))
                for object_elt in ann_tree.getroot().findall('object'):

                    f_class = object_elt.find('name').text
                    c_class = f_class[:-1]
                    bbox_elt = object_elt.find('bndbox')
                    bbox_xyxy_tags = ['xmin', 'ymin', 'xmax', 'ymax']
                    bbox_xyxy = [int(bbox_elt.find(tag).text) for tag in bbox_xyxy_tags]
                    image_path = os.path.join(
                        self.cluttered_image_dir, scene_prefix, f'{base_file_name}.jpg')
                    example = {
                        IMAGE_PATH_COLUMN_NAME: image_path,
                        LABEL_COLUMN_NAME: self.classes.index(f_class),
                        COARSE_LABEL_COLUMN_NAME: self.coarse_classes.index(c_class),
                        BOUNDING_BOX_COLUMN_NAME: bbox_xyxy,
                        SUPPORT_BOOL_COLUMN_NAME: False,
                        CLUTTERED_BOOL_COLUMN_NAME: True
                    }
                    yield (ex_id, example)
                    ex_id += 1


def generate_icb_episodes(dataset, episode_num=None, n_way=1, seed=None, cluttered_scenes=False):

    if seed is not None:
        random.seed(seed)

    labels = dataset[LABEL_COLUMN_NAME]
    is_support_column = dataset[SUPPORT_BOOL_COLUMN_NAME]
    is_cluttered_column = dataset[CLUTTERED_BOOL_COLUMN_NAME]
    icb_classes = {}
    for row_ind in range(len(labels)):
        lab = labels[row_ind]
        if lab not in icb_classes:
            icb_classes[lab] = {'support': [], 'query': []}

        if is_support_column[row_ind]:
            icb_classes[lab]['support'].append(row_ind)
        elif cluttered_scenes == is_cluttered_column[row_ind]:
            icb_classes[lab]['query'].append(row_ind)

    lab_list = list(icb_classes.keys())
    for lab in lab_list:
        if len(icb_classes[lab]['query']) == 0:
            icb_classes.pop(lab)
    lab_list = list(icb_classes.keys())
    episodes = []
    for _ in range(episode_num):
        if n_way is None:
            chosen_classes = lab_list
        else:
            chosen_classes = random.sample(lab_list, n_way)
        support_inds = []
        query_inds = []
        for ch_cls in chosen_classes:
            support_inds.extend(icb_classes[ch_cls]['support'])
            query_inds.extend(icb_classes[ch_cls]['query'])
        episodes.append({'support': support_inds, 'query': query_inds})
    return episodes