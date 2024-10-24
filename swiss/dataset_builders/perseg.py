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
import random


# Schema for the few-shot segmentation dataset
IMAGE_PATH_COLUMN_NAME = "image_path"
LABEL_COLUMN_NAME = "labels"
SEGMENTATION_PATH_COLUMN_NAME = "segmentation_path"
SUPPORT_BOOL_COLUMN_NAME = "is_support_image"


# Name of the dataset usually match the script name with CamelCase instead of snake_case
class PerSegDatasetBuilder(datasets.GeneratorBasedBuilder):

    BUILDER_CONFIG_CLASS = datasets.BuilderConfig

    def __init__(self, *args, **kwargs):

        self.dataset_name = "PerSeg"
        if 'config_name' not in kwargs.keys():
            kwargs['config_name'] = 'default'
        dataset_cache_dir = os.path.join(os.getcwd(), "datasets_cache")
        kwargs['cache_dir'] = dataset_cache_dir

        super().__init__(*args, **kwargs)

        self.val_dir = "Images"
        self.annotation_dir = "Annotations"
        self.data_dir = self.config.data_dir
        self.classes = os.listdir(os.path.join(self.data_dir, self.val_dir))

        self.download_and_prepare()


    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset

        feature_dict = {
            IMAGE_PATH_COLUMN_NAME: datasets.Value("string"),
            LABEL_COLUMN_NAME: datasets.Value("int32"),
            SEGMENTATION_PATH_COLUMN_NAME: datasets.Value("string"),
            SUPPORT_BOOL_COLUMN_NAME: datasets.Value("bool"),
        }

        features = datasets.Features(feature_dict)
        return datasets.DatasetInfo(features=features)


    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        dataset_val = datasets.SplitGenerator(
            name=datasets.Split.VALIDATION,
            # These kwargs will be passed to _generate_examples
            gen_kwargs={
                "segmentation_ann_dir": os.path.join(self.data_dir, self.annotation_dir),
                "image_dir": os.path.join(self.data_dir, self.val_dir),
            },
        )

        splits = [dataset_val]

        return splits


    def _generate_examples(
        # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
        self, segmentation_ann_dir, image_dir
    ):
        """ Yields examples as (key, example) tuples. """
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is here for legacy reason (tfds) and is not important in itself.

        class_list = os.listdir(image_dir)

        ex_id = 0
        for class_name in class_list:
            class_dir = os.path.join(image_dir, class_name)
            seg_ann_class_dir = os.path.join(segmentation_ann_dir, class_name)
            is_support = True
            for image_file_name in sorted(os.listdir(class_dir)):
                image_path = os.path.join(class_dir, image_file_name)
                seg_ann_path = os.path.join(seg_ann_class_dir, image_file_name.replace('.jpg', '.png'))
                example = {
                    IMAGE_PATH_COLUMN_NAME: image_path,
                    LABEL_COLUMN_NAME: self.classes.index(class_name),
                    SEGMENTATION_PATH_COLUMN_NAME: seg_ann_path,
                    SUPPORT_BOOL_COLUMN_NAME: is_support,
                }
                yield (ex_id, example)
                is_support = False
                ex_id += 1


def generate_perseg_episodes(dataset, episode_num=1, n_way=1, seed=None):

    if seed is not None:
        random.seed(seed)

    labels = dataset[LABEL_COLUMN_NAME]
    is_support_column = dataset[SUPPORT_BOOL_COLUMN_NAME]
    icb_classes = {}
    for row_ind in range(len(labels)):
        lab = labels[row_ind]
        if lab not in icb_classes:
            icb_classes[lab] = {'support': [], 'query': []}

        if is_support_column[row_ind]:
            icb_classes[lab]['support'].append(row_ind)
        else:
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
