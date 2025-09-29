"""
@date: 2021/7/19
@description:
"""

from torch.nn import L1Loss
from .led_loss import LEDLoss
from .grad_loss import GradLoss
from .boundary_loss import BoundaryLoss
from .object_loss import ObjectLoss, HeatmapLoss
