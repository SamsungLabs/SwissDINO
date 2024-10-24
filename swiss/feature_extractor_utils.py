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
import torch


def generate_features_dino(images, feature_extractor_model, image_transforms, batch_size=64, device='cuda'):
    pixel_values = torch.stack([image_transforms(img) for img in images]).to(device)
    batch_num = (len(images)-1) // batch_size + 1
    batches = [pixel_values[batch_ind*batch_size:(batch_ind+1)*batch_size] for batch_ind in range(batch_num)]
    hidden_states_map_batches = []
    class_tokens_batches = []
    for batch in batches:
        with torch.no_grad():
            output_tensors = feature_extractor_model.get_intermediate_layers(batch, reshape=True, return_class_token=True)[0]
        hidden_states_map_batches.append(output_tensors[0].permute(0,2,3,1).cpu().numpy())  # Bx32x32x768
        class_tokens_batches.append(output_tensors[1].cpu().numpy())
    return np.vstack(hidden_states_map_batches), np.vstack(class_tokens_batches)


def compare_feature_maps(query_feature_map, query_mask, support_prototype):

    if np.sum(query_mask)==0:
        return 0

    feat_dim = query_feature_map.shape[2]
    support_prototype = support_prototype / np.linalg.norm(support_prototype)
    query_prototype = np.mean(query_feature_map[query_mask].reshape(-1, feat_dim), axis=0)
    query_prototype = query_prototype / np.linalg.norm(query_prototype)

    return query_prototype @ support_prototype.T