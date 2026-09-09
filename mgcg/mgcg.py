"""
Multi-Granularity Conditional Guidance (MGCG) network.

The MGCG network produces three-dimensional conditional guidance priors
(y_global, y_local, y_texture) for the diffusion classification network:

  - Global branch:   CAM-based lesion localization + top-K aggregation  (Eq. 1)
  - Local branch:    RPN-guided ROI cropping + attention MIL pooling    (Eq. 2-3)
  - Texture branch:  multi-scale convolutional texture extraction with
                     residual blocks and channel attention              (Eq. 4)

The fused prediction y_fusion is used as the prior mean of the reverse
diffusion process at timestep T.
"""
import torch
import torch.nn as nn

from .modules import (
    GlobalNetwork,
    TopTPercentAggregationFunction,
    RetrieveROIModule,
    LocalNetwork,
    AttentionModule,
)
from .texture import TextureNetwork
from . import tools


class MGCG(nn.Module):
    def __init__(self, config):
        super(MGCG, self).__init__()

        # resolve the device type from the runtime environment so that the
        # network also works on CPU-only machines
        import torch as _torch
        _device_type = "gpu" if _torch.cuda.is_available() else "cpu"

        self.experiment_parameters = {
            "device_type": _device_type,
            "gpu_number": 0,
            # model related hyper-parameters
            "cam_size": (7, 7),
            "K": 6,
            "crop_shape": (32, 32),
            "post_processing_dim": 512,
            "num_classes": config.data.num_classes,
            "use_v1_global": True,
            "percent_t": 1.0,
            "in_channels": getattr(config.data, "num_channels", 3),
        }
        self.cam_size = self.experiment_parameters["cam_size"]

        # construct networks
        # global network
        self.global_network = GlobalNetwork(self.experiment_parameters, self)
        self.global_network.add_layers()

        # aggregation function
        self.aggregation_function = TopTPercentAggregationFunction(
            self.experiment_parameters, self)

        # detection module
        self.retrieve_roi_crops = RetrieveROIModule(
            self.experiment_parameters, self)

        # detection network
        self.local_network = LocalNetwork(self.experiment_parameters, self)
        self.local_network.add_layers()

        # MIL module
        self.attention_module = AttentionModule(
            self.experiment_parameters, self)
        self.attention_module.add_layers()

        # multi-scale texture branch (Eq. 4)
        self.texture_network = TextureNetwork(self.experiment_parameters)

    def _convert_crop_position(self, crops_x_small, cam_size, x_original):
        """
        Function that converts the crop locations from cam_size to x_original.
        """
        # retrieve the dimension of both the original image and the small version
        h, w = cam_size
        _, _, H, W = x_original.size()

        # interpolate the 2d index in h_small to index in x_original
        top_k_prop_x = crops_x_small[:, :, 0] / h
        top_k_prop_y = crops_x_small[:, :, 1] / w
        # sanity check
        assert np_max(top_k_prop_x) <= 1.0, "top_k_prop_x >= 1.0"
        assert np_min(top_k_prop_x) >= 0.0, "top_k_prop_x <= 0.0"
        assert np_max(top_k_prop_y) <= 1.0, "top_k_prop_y <= 1.0"
        assert np_min(top_k_prop_y) >= 0.0, "top_k_prop_y <= 0.0"
        # interpolate the crop position from cam_size to x_original
        top_k_interpolate_x = torch.unsqueeze(torch.round(top_k_prop_x * H), -1)
        top_k_interpolate_y = torch.unsqueeze(torch.round(top_k_prop_y * W), -1)
        top_k_interpolate_2d = torch.cat(
            [top_k_interpolate_x, top_k_interpolate_y], dim=-1)
        return top_k_interpolate_2d

    def _retrieve_crop(self, x_original_pytorch, crop_positions, crop_method):
        """
        Function that takes in the original image and cropping position and returns the crops.
        """
        batch_size, num_crops, _ = crop_positions.shape
        crop_h, crop_w = self.experiment_parameters["crop_shape"]

        output = torch.ones((batch_size, num_crops, crop_h, crop_w))
        if self.experiment_parameters["device_type"] == "gpu":
            device = torch.device(
                "cuda:{}".format(self.experiment_parameters["gpu_number"]))
            output = output.to(device)
        for i in range(batch_size):
            for j in range(num_crops):
                tools.crop_pytorch(x_original_pytorch[i, 0, :, :],
                                   self.experiment_parameters["crop_shape"],
                                   crop_positions[i, j, :],
                                   output[i, j, :, :],
                                   method=crop_method)
        return output

    def forward(self, x_original):
        """
        :param x_original: (N, C, H, W) input pathology image batch.
        :return:
            y_fusion:  fused prediction prior (mean of the three branches),
            y_global:  global guidance prior     (Eq. 1),
            y_local:   local guidance prior      (Eq. 3),
            y_texture: texture guidance prior    (Eq. 4).
        """
        # global network: x_small -> class activation map
        h_g, self.saliency_map = self.global_network.forward(x_original)

        # calculate y_global (Eq. 1)
        self.y_global = self.aggregation_function.forward(self.saliency_map)

        # region proposal network
        small_x_locations = self.retrieve_roi_crops.forward(
            x_original, self.cam_size, self.saliency_map)

        # convert crop locations that is on self.cam_size to x_original
        self.patch_locations = self._convert_crop_position(
            small_x_locations, self.cam_size, x_original)

        # patch retriever
        crops_variable = self._retrieve_crop(
            x_original, self.patch_locations, self.retrieve_roi_crops.crop_method)
        self.patches = crops_variable.data.cpu().numpy()

        # local network + attention MIL (Eq. 2-3)
        batch_size, num_crops, I, J = crops_variable.size()
        crops_variable = crops_variable.view(
            batch_size * num_crops, I, J).unsqueeze(1)
        h_crops = self.local_network.forward(crops_variable).view(
            batch_size, num_crops, -1)

        z, self.patch_attns, self.y_local = self.attention_module.forward(h_crops)

        # multi-scale texture branch on the whole image (Eq. 4)
        self.y_texture = self.texture_network.forward(x_original)

        # fused prior used as the mean of the diffusion prior at timestep T
        self.y_fusion = (self.y_global + self.y_local + self.y_texture) / 3.0
        return self.y_fusion, self.y_global, self.y_local, self.y_texture


def np_max(x):
    import numpy as np
    if isinstance(x, torch.Tensor):
        return np.max(x.detach().cpu().numpy())
    return np.max(x)


def np_min(x):
    import numpy as np
    if isinstance(x, torch.Tensor):
        return np.min(x.detach().cpu().numpy())
    return np.min(x)
