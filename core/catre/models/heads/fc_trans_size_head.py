import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.cnn import normal_init, constant_init
from lib.torch_utils.layers.layer_utils import get_norm, get_nn_act_func
from lib.torch_utils.layers.conv_module import ConvModule


class FC_TransSizeHead(nn.Module):
    def __init__(
        self,
        in_dim=1024,
        feat_dim=256,
        num_layers=2,
        rot_dim=4,
        norm="none",
        num_gn_groups=32,
        act="leaky_relu",
        num_classes=1,
        norm_input=False,
        dropout=False,
    ):
        """
        rot_dim: 4 for quaternion, 6 for rot6d
        num_classes: default 1 (either single class or class-agnostic)
        """
        super().__init__()
        self.norm = get_norm(norm, feat_dim, num_gn_groups=num_gn_groups)
        self.act_func = act_func = get_nn_act_func(act)
        self.num_classes = num_classes
        self.rot_dim = rot_dim

        self.linears = nn.ModuleList()
        if norm_input:
            self.linears.append(nn.BatchNorm1d(in_dim))
        for _i in range(num_layers):
            _in_dim = in_dim if _i == 0 else feat_dim
            self.linears.append(nn.Linear(_in_dim, feat_dim))
            self.linears.append(get_norm(norm, feat_dim, num_gn_groups=num_gn_groups))
            self.linears.append(act_func)
            if dropout:
                self.linears.append(nn.Dropout(p=0.5))

        self.fc_t = nn.Linear(feat_dim, 3 * num_classes)
        self.fc_s = nn.Linear(feat_dim, 3 * num_classes)

        # init ------------------------------------
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Conv1d)):
                normal_init(m, std=0.001)
            elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.Linear):
                normal_init(m, std=0.001)
        normal_init(self.fc_t, std=0.01)
        normal_init(self.fc_s, std=0.01)

    def forward(self, x):
        """
        x: should be flattened
        """
        for _layer in self.linears:
            x = _layer(x)

        trans = self.fc_t(x)
        scale = self.fc_s(x)
        return trans, scale
