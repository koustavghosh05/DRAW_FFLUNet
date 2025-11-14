import torch
import torch.nn.functional as nnfx
from torch import nn

from nnunetv2.training.nnUNetTrainer.variants.fflunet.blocks import (
    DepthWiseSeparableConvolution,
    DownSample,
    Input,
    DynamicMultiViewFeatureFusion,
    Output,
    UpSample,
)
from nnunetv2.training.nnUNetTrainer.variants.network_architecture.nnUNetTrainerNoDeepSupervision import (
    nnUNetTrainerNoDeepSupervision,
)
from nnunetv2.utilities.plans_handling.plans_handler import (
    ConfigurationManager,
    PlansManager,
)

class FFLUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.input = Input(in_channels, 32)

        self.m1 = DynamicMultiViewFeatureFusion(32, 32)
        self.d1 = DownSample(32, 64)

        self.m2 = DynamicMultiViewFeatureFusion(64, 64)
        self.d2 = DownSample(64, 128)

        self.m3 = DynamicMultiViewFeatureFusion(128, 128)
        self.d3 = DownSample(128, 256)

        self.m4 = DynamicMultiViewFeatureFusion(256, 256)
        self.d4 = DownSample(256, 320)

        self.u4 = UpSample(320, 256)
        self.s4 = DepthWiseSeparableConvolution(256, 256)

        self.u3 = UpSample(256, 128)
        self.s3 = DepthWiseSeparableConvolution(128, 128)

        self.u2 = UpSample(128, 64)
        self.s2 = DepthWiseSeparableConvolution(64, 64)

        self.u1 = UpSample(64, 32)
        self.s1 = DepthWiseSeparableConvolution(32, 32)

        self.w4 = nn.Parameter(torch.rand(2), requires_grad=True)
        self.w3 = nn.Parameter(torch.rand(2), requires_grad=True)
        self.w2 = nn.Parameter(torch.rand(2), requires_grad=True)
        self.w1 = nn.Parameter(torch.rand(2), requires_grad=True)

        self.output = Output(32, out_channels)

    def forward(self, x):
        x = self.input(x)

        x = x1 = self.m1(x, 4)
        x = self.d1(x)

        x = x2 = self.m2(x, 4)
        x = self.d2(x)

        x = x3 = self.m3(x, 4)
        x = self.d3(x)

        x = x4 = self.m4(x, 4)
        x = self.d4(x)

        w4 = nnfx.softmax(self.w4, dim=0)
        w3 = nnfx.softmax(self.w3, dim=0)
        w2 = nnfx.softmax(self.w2, dim=0)
        w1 = nnfx.softmax(self.w1, dim=0)

        # print("****u4 shape: ", self.u4(x).shape, " s4 shape: ", self.s4(x4).shape)
        x = w4[0] * self.u4(x) + w4[1] * self.s4(x4)

        # print("****u3 shape: ", self.u3(x).shape, " s3 shape: ", self.s3(x3).shape)
        x = w3[0] * self.u3(x) + w3[1] * self.s3(x3)

        # print("****u2 shape: ", self.u2(x).shape, " s2 shape: ", self.s2(x2).shape)
        x = w2[0] * self.u2(x) + w2[1] * self.s2(x2)

        # print("****u1 shape: ", self.u1(x).shape, " s1 shape: ", self.s1(x1).shape)
        x = w1[0] * self.u1(x) + w1[1] * self.s1(x1)
       

        x = self.output(x)
        return x


class nnUNetTrainer_FFLUNet(nnUNetTrainerNoDeepSupervision):  
    @staticmethod
    def build_network_architecture(
        plans_manager: PlansManager,
        dataset_json,
        configuration_manager: ConfigurationManager,
        num_input_channels,
        enable_deep_supervision: bool = False,
    ) -> nn.Module:
        label_manager = plans_manager.get_label_manager(dataset_json)
        num_op_channels = label_manager.num_segmentation_heads
        return FFLUNet(
            in_channels=num_input_channels,
            out_channels=num_op_channels,
        )
