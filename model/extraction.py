"""
Full definition of a Extraction model, all of it in this single file.
Ref:
1) the official Darknet implementation:
https://github.com/pjreddie/darknet/blob/master/examples/classifier.c
https://github.com/pjreddie/darknet/blob/master/cfg/extraction.cfg
"""

import math
import inspect
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Tuple

import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class ExtractionConfig:
    img_h: int = 224
    img_w: int = 224
    n_class: int = 1000


class ExtractionConv2d(nn.Module):
    """
    A Conv2d layer with a BarchNorm2d and a LeakyReLU activation.
    """
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        # Darknet implementation uses bias=False when batch norm is used.
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=1e-06, momentum=0.01)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.leaky_relu(x, 0.1, inplace=True)


class ExtractionBackbone(nn.Module):
    """
    Backbone of the Extraction model, i.e., first 20th conv layers of YOLOv1.
    """
    def __init__(self, config: ExtractionConfig) -> None:
        super().__init__()
        self.conv1 = ExtractionConv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(2, stride=2)

        self.conv2 = ExtractionConv2d(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(2, stride=2)

        self.conv3 = ExtractionConv2d(192, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = ExtractionConv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv5 = ExtractionConv2d(256, 256, kernel_size=1, stride=1, padding=0)
        self.conv6 = ExtractionConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.maxpool3 = nn.MaxPool2d(2, stride=2)

        self.conv7 = ExtractionConv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv8 = ExtractionConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv9 = ExtractionConv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv10 = ExtractionConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv11 = ExtractionConv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv12 = ExtractionConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv13 = ExtractionConv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.conv14 = ExtractionConv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv15 = ExtractionConv2d(512, 512, kernel_size=1, stride=1, padding=0)
        self.conv16 = ExtractionConv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.maxpool4 = nn.MaxPool2d(2, stride=2)

        self.conv17 = ExtractionConv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv18 = ExtractionConv2d(512, 1024, kernel_size=3, stride=1, padding=1)
        self.conv19 = ExtractionConv2d(1024, 512, kernel_size=1, stride=1, padding=0)
        self.conv20 = ExtractionConv2d(512, 1024, kernel_size=3, stride=1, padding=1)


    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): size(N, 3, img_h, img_w)
        Returns:
            x (Tensor): (N, 1024, img_h / 224 * 7, img_w / 224 * 7)
        """
        # N x 3 x 224 x 224
        x = self.conv1(x)
        # N x 64 x 112 x 112
        x = self.maxpool1(x)
        # N x 64 x 56 x 56

        x = self.conv2(x)
        # N x 192 x 56 x 56
        x = self.maxpool2(x)
        # N x 192 x 28 x 28

        x = self.conv3(x)
        # N x 128 x 28 x 28
        x = self.conv4(x)
        # N x 256 x 28 x 28
        x = self.conv5(x)
        # N x 256 x 28 x 28
        x = self.conv6(x)
        # N x 512 x 28 x 28
        x = self.maxpool3(x)
        # N x 512 x 14 x 14

        x = self.conv7(x)
        # N x 256 x 14 x 14
        x = self.conv8(x)
        # N x 512 x 14 x 14
        x = self.conv9(x)
        # N x 256 x 14 x 14
        x = self.conv10(x)
        # N x 512 x 14 x 14
        x = self.conv11(x)
        # N x 256 x 14 x 14
        x = self.conv12(x)
        # N x 512 x 14 x 14
        x = self.conv13(x)
        # N x 256 x 14 x 14
        x = self.conv14(x)
        # N x 512 x 14 x 14
        x = self.conv15(x)
        # N x 512 x 14 x 14
        x = self.conv16(x)
        # N x 1024 x 14 x 14
        x = self.maxpool4(x)
        # N x 1024 x 7 x 7

        x = self.conv17(x)
        # N x 512 x 7 x 7
        x = self.conv18(x)
        # N x 1024 x 7 x 7
        x = self.conv19(x)
        # N x 512 x 7 x 7
        x = self.conv20(x)
        # N x 1024 x 7 x 7

        return x


class Extraction(nn.Module):
    def __init__(self, config: ExtractionConfig) -> None:
        super().__init__()
        self.config = config

        self.backbone = ExtractionBackbone(config)

        self.head = nn.Sequential(
            ExtractionConv2d(1024, 1000, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1000, config.n_class)
        )


    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        """
        n_params = sum(p.numel() for p in self.parameters())
        return n_params


    def _init_weights(self, module):
        raise NotImplementedError("FUTURE: init weights for Extraction model")


    def forward(self, imgs: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Args:
            imgs (Tensor): size(N, 3, img_h, img_w)
            targets (Tensor): size(N, n_class)
        Returns:
            logits (Tensor): size(N,)
            loss (Tensor): size(,)
        """
        device = imgs.device

        # Forward the Extraction model itself
        # N x 3 x 224 x 224
        x = self.backbone(imgs)
        # N x 1024 x 7 x 7
        logits = self.head(x)
        # N x n_class

        if targets is not None:
            # If we are given some desired targets also calculate the loss
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss


    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        raise NotImplementedError("FUTURE: init from pretrained model")


    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type, use_fused):
        # Start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # Filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim groups. any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls decay, all biases and norms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        if use_fused:
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer


    def estimate_tops(self):
        """
        Estimate the number of TOPS and parameters in the model.
        """
        raise NotImplementedError("FUTURE: estimate TOPS for Extraction model")


    @torch.no_grad()
    def generate(self, imgs, top_k=None):
        """
        Predict on test imgs and return the top_k predictions.
        """
        # Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        self.eval()
        raise NotImplementedError("FUTURE: generate for Extraction model")
        self.train()



if __name__ == '__main__':
    # Test the model by `python -m model.extraction` from the workspace directory
    config = ExtractionConfig()
    # config = ExtractionConfig(img_h=448, img_w=448)
    model = Extraction(config)
    print(model)
    print(f"num params: {model.get_num_params():,}")

    imgs = torch.randn(2, 3, config.img_h, config.img_w)
    targets = torch.randint(0, config.n_class, (2,))
    logits, loss = model(imgs, targets)
    print(f"logits shape: {logits.shape}")
    if loss is not None:
        print(f"loss shape: {loss.shape}")
