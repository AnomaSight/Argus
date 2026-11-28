"""Transcription of the official PatchCore feature extraction, used only to test A1.3.

Source: https://github.com/amazon-science/patchcore-inspection at commit
fcaa92f124fb1ad74a7acf56726decd4b27cbcad
  - src/patchcore/patchcore.py: PatchMaker.patchify, and the layer loop and the
    preprocessing / preadapt_aggregator calls of PatchCore._embed
    https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/patchcore.py
  - src/patchcore/common.py: Preprocessing, MeanMapper, Aggregator, NetworkFeatureAggregator,
    ForwardHook, LastLayerToExtractReachedException
    https://github.com/amazon-science/patchcore-inspection/blob/fcaa92f124fb1ad74a7acf56726decd4b27cbcad/src/patchcore/common.py

Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
except in compliance with the License. You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0. Unless required by applicable law or agreed to in
writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.

Changes from the original: faiss, numpy, the sampler, scoring and everything not needed to
produce embeddings are left out; the device handling of NetworkFeatureAggregator is dropped;
_embed's loop is a function taking the layer tensors (``embed_layers``), so both
implementations can be fed the same tensors. The arithmetic of every kept line is unchanged.
If this file and the original ever disagree, this file is wrong.
"""

import copy

import torch
import torch.nn.functional as F


class PatchMaker:
    def __init__(self, patchsize, stride=None):
        self.patchsize = patchsize
        self.stride = stride

    def patchify(self, features, return_spatial_info=False):
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (s + 2 * padding - 1 * (self.patchsize - 1) - 1) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features


class Preprocessing(torch.nn.Module):
    def __init__(self, input_dims, output_dim):
        super().__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        self.preprocessing_modules = torch.nn.ModuleList()
        for _input_dim in input_dims:
            module = MeanMapper(output_dim)
            self.preprocessing_modules.append(module)

    def forward(self, features):
        _features = []
        for module, feature in zip(self.preprocessing_modules, features, strict=False):
            _features.append(module(feature))
        return torch.stack(_features, dim=1)


class MeanMapper(torch.nn.Module):
    def __init__(self, preprocessing_dim):
        super().__init__()
        self.preprocessing_dim = preprocessing_dim

    def forward(self, features):
        features = features.reshape(len(features), 1, -1)
        return F.adaptive_avg_pool1d(features, self.preprocessing_dim).squeeze(1)


class Aggregator(torch.nn.Module):
    def __init__(self, target_dim):
        super().__init__()
        self.target_dim = target_dim

    def forward(self, features):
        """Returns reshaped and average pooled features."""
        # batchsize x number_of_layers x input_dim -> batchsize x target_dim
        features = features.reshape(len(features), 1, -1)
        features = F.adaptive_avg_pool1d(features, self.target_dim)
        return features.reshape(len(features), -1)


class LastLayerToExtractReachedException(Exception):
    pass


class ForwardHook:
    def __init__(self, hook_dict, layer_name: str, last_layer_to_extract: str):
        self.hook_dict = hook_dict
        self.layer_name = layer_name
        self.raise_exception_to_break = copy.deepcopy(layer_name == last_layer_to_extract)

    def __call__(self, module, input, output):
        self.hook_dict[self.layer_name] = output
        if self.raise_exception_to_break:
            raise LastLayerToExtractReachedException()
        return None


class NetworkFeatureAggregator(torch.nn.Module):
    """Runs a full torchvision network up to the last requested layer, via forward hooks."""

    def __init__(self, backbone, layers_to_extract_from):
        super().__init__()
        self.layers_to_extract_from = layers_to_extract_from
        self.backbone = backbone
        if not hasattr(backbone, "hook_handles"):
            self.backbone.hook_handles = []
        for handle in self.backbone.hook_handles:
            handle.remove()
        self.outputs = {}

        for extract_layer in layers_to_extract_from:
            forward_hook = ForwardHook(self.outputs, extract_layer, layers_to_extract_from[-1])
            network_layer = backbone.__dict__["_modules"][extract_layer]
            if isinstance(network_layer, torch.nn.Sequential):
                self.backbone.hook_handles.append(
                    network_layer[-1].register_forward_hook(forward_hook)
                )
            else:
                self.backbone.hook_handles.append(network_layer.register_forward_hook(forward_hook))

    def forward(self, images):
        self.outputs.clear()
        with torch.no_grad():
            try:
                _ = self.backbone(images)
            except LastLayerToExtractReachedException:
                pass
        return self.outputs


def embed_layers(features, patchsize=3, patchstride=1, pretrain_dim=1024, target_dim=1024):
    """PatchCore._embed from the list of layer tensors on: returns (B * H * W, target_dim)."""
    patch_maker = PatchMaker(patchsize, stride=patchstride)
    preprocessing = Preprocessing([f.shape[1] for f in features], pretrain_dim)
    preadapt_aggregator = Aggregator(target_dim=target_dim)

    features = [patch_maker.patchify(x, return_spatial_info=True) for x in features]
    patch_shapes = [x[1] for x in features]
    features = [x[0] for x in features]
    ref_num_patches = patch_shapes[0]

    for i in range(1, len(features)):
        _features = features[i]
        patch_dims = patch_shapes[i]

        _features = _features.reshape(
            _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
        )
        _features = _features.permute(0, -3, -2, -1, 1, 2)
        perm_base_shape = _features.shape
        _features = _features.reshape(-1, *_features.shape[-2:])
        _features = F.interpolate(
            _features.unsqueeze(1),
            size=(ref_num_patches[0], ref_num_patches[1]),
            mode="bilinear",
            align_corners=False,
        )
        _features = _features.squeeze(1)
        _features = _features.reshape(*perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1])
        _features = _features.permute(0, -2, -1, 1, 2, 3)
        _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
        features[i] = _features
    features = [x.reshape(-1, *x.shape[-3:]) for x in features]

    features = preprocessing(features)
    features = preadapt_aggregator(features)
    return features, patch_shapes


def official_embed(net, images, layers=("layer2", "layer3")):
    """The whole official path on a full torchvision network: hooks, then _embed."""
    net.eval()
    with torch.no_grad():
        outputs = NetworkFeatureAggregator(net, list(layers))(images)
        return embed_layers([outputs[layer] for layer in layers])
