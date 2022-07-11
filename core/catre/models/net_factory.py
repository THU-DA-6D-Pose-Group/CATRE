from .pointnets.pointnet import PointNetfeat

from .heads.fc_trans_size_head import FC_TransSizeHead
from .heads.conv_out_per_rot_head import ConvOutPerRotHead

PCLNETS = {
    "point_net": PointNetfeat,
}

HEADS = {
    "FC_TransSizeHead": FC_TransSizeHead,
    "ConvOutPerRotHead": ConvOutPerRotHead,
}
