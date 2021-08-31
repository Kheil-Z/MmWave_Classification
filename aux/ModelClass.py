import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

#
# class model_knife(nn.Module):
#     def __init__(self):
#         super(model_knife, self).__init__()
#
#         self.CNN = nn.Sequential(
#             # In : (1,63,63)
#             nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#
#             # In : (8,30,30)
#             nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#
#             # In : (16,14,14)
#             nn.Conv2d(16, 8, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#         )
#         self.FC = nn.Sequential(
#             # In : (batch_size,16*6*6)
#             # nn.Linear(8*6*6, 128),
#             # nn.ReLU(),
#             nn.Linear(8 * 6 * 6, 4),
#             # nn.Softmax(),
#         )
#
#     def forward(self, x):
#         inter = self.CNN(x)  # torch.cat([x,label],1)).view(-1,1).squeeze(1)
#
#         out = self.FC(inter.view(x.shape[0], -1))
#         return (out)
#
#
# class model_knife_spoon(nn.Module):
#     def __init__(self):
#         super(model_knife_spoon, self).__init__()
#
#         self.CNN = nn.Sequential(
#             # In : (1,63,63)
#             nn.Conv2d(1, 8, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#
#             # In : (8,30,30)
#             nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#
#             # In : (16,14,14)
#             nn.Conv2d(16, 8, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#         )
#         self.FC = nn.Sequential(
#             # In : (batch_size,16*6*6)
#             # nn.Linear(8*6*6, 128),
#             # nn.ReLU(),
#             nn.Linear(8 * 6 * 6, 3),
#             # nn.Softmax(),
#         )
#
#     def forward(self, x):
#         inter = self.CNN(x)  # torch.cat([x,label],1)).view(-1,1).squeeze(1)
#
#         out = self.FC(inter.view(x.shape[0], -1))
#         return (out)
#
#
# class model_knife_spoon_big(nn.Module):
#     def __init__(self):
#         super(model_knife_spoon_big, self).__init__()
#
#         self.CNN = nn.Sequential(
#             # In : (1,63,63)
#             nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#
#             # In : (8,30,30)
#             nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#
#             # In : (16,14,14)
#             nn.Conv2d(32, 16, kernel_size=4, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#         )
#         self.FC = nn.Sequential(
#             # In : (batch_size,16*6*6)
#             nn.Linear(16 * 6 * 6, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3),
#             # nn.Softmax(),
#         )
#
#     def forward(self, x):
#         inter = self.CNN(x)  # torch.cat([x,label],1)).view(-1,1).squeeze(1)
#
#         out = self.FC(inter.view(x.shape[0], -1))
#         return (out)
#
#
# class model_knife_spoon_ROI(nn.Module):
#     def __init__(self):
#         super(model_knife_spoon_ROI, self).__init__()
#
#         self.CNN = nn.Sequential(
#             # In : (1,32,32)
#             nn.Conv2d(1, 8, kernel_size=2, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#
#             # In : (8,16,16)
#             nn.Conv2d(8, 16, kernel_size=2, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#
#             # In : (16,8,8)
#             nn.Conv2d(16, 8, kernel_size=2, stride=2, padding=0, bias=True),
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#         )
#         self.FC = nn.Sequential(
#             # In : (batch_size,8*4*4)
#             nn.Linear(8 * 4 * 4, 3),
#             # nn.ReLU(),
#             # nn.Linear(128, 3),
#             # nn.Softmax(),
#         )
#
#     def forward(self, x):
#         inter = self.CNN(x)  # torch.cat([x,label],1)).view(-1,1).squeeze(1)
#
#         out = self.FC(inter.view(x.shape[0], -1))
#         return (out)
# class model_VGGNet_32(nn.Module):
#     def __init__(self):
#         super(model_VGGNet_32, self).__init__()
#         self.features = nn.Sequential(
#             # In : (1,32,32)
#             nn.Conv2d(1, 8, 3, padding=1),  # Conv1
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Conv2d(8, 8, 3, padding=1),  # Conv2
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Pool1
#
#              # In : (8,16,16)
#             nn.Conv2d(8, 16, 3, padding=1),  # Conv3
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, 3, padding=1),  # Conv4
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Pool2
#
#             # In : (16,8,8)
#             nn.Conv2d(16, 32, 3, padding=1),  # Conv5
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),  # Conv6
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Pool3
#
#             # In : (32,4,4)
#             nn.Conv2d(32, 64, 3, padding=1),  # Conv7
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),  # Conv8
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Pool4
#
#             # In : (64,2,2)
#             nn.Conv2d(64, 64, 3, padding=1),  # Conv9
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),  # Conv10
#             nn.BatchNorm2d(64),
#             nn.ReLU()
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(2 * 2 * 64, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3)
#         )
#
#     def forward(self, x):
#       inter = self.features(x)
#       out = self.classifier(inter.view(x.shape[0],-1))
#       return out
#
#
# class model_CNN4_32(nn.Module):
#     def __init__(self):
#         super(model_CNN4_32, self).__init__()
#
#         self.features = nn.Sequential(
#             # In : (1,32,32)
#             nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.ReLU(),
#             # In : (16,30,30)
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#
#             # In : (32,14,14)
#             nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             # In : (32,12,12)
#             nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.classifier = nn.Sequential(
#             # In : (batch_size,16*6*6)
#             nn.Linear(16 * 5 * 5, 128),
#             nn.ReLU(),
#             nn.Linear(128, 3)
#         )
#
#     def forward(self, x):
#         inter = self.features(x)
#         out = self.classifier(inter.view(x.shape[0], -1))
#         return (out)
#
#
# class model_VGGNet_32_Voxel_dropout(nn.Module):
#     def __init__(self):
#         super(model_VGGNet_32_Voxel_dropout, self).__init__()
#         self.features_heatmap = nn.Sequential(
#             # In : (1,32,32)
#             nn.Conv2d(1, 8, 3, padding=1),  # Conv1
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.Conv2d(8, 8, 3, padding=1),  # Conv2
#             nn.BatchNorm2d(8),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Pool1
#
#             # In : (8,16,16)
#             nn.Conv2d(8, 16, 3, padding=1),  # Conv3
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.Conv2d(16, 16, 3, padding=1),  # Conv4
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Pool2
#
#             # In : (16,8,8)
#             nn.Conv2d(16, 32, 3, padding=1),  # Conv5
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.Conv2d(32, 32, 3, padding=1),  # Conv6
#             nn.BatchNorm2d(32),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Pool3
#
#             # In : (32,4,4)
#             nn.Conv2d(32, 64, 3, padding=1),  # Conv7
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 64, 3, padding=1),  # Conv8
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2, 2),  # Pool4
#
#             # In : (64,2,2)
#             nn.Conv2d(64, 64, 3, padding=1),  # Conv9
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 32, 3, padding=1),  # Conv10
#             nn.BatchNorm2d(32),
#             nn.ReLU()
#         )
#         self.features_voxels = nn.Sequential(
#             # In : (1,26,26,26)
#             nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.ReLU(),
#             # In : (16,24,24,24)
#             nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.BatchNorm3d(32),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2),
#
#             # In : (32,11,11,11)
#             nn.Conv3d(32, 16, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.BatchNorm3d(16),
#             nn.ReLU(),
#             # In : (32,9,9,9)
#             nn.Conv3d(16, 8, kernel_size=3, stride=1, padding=0, bias=True),
#             nn.ReLU(),
#             nn.MaxPool3d(kernel_size=2, stride=2)
#
#             # Out : (8,3,3,3)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear(2 * 2 * 32 + 8 * 3 * 3 * 3, 128),
#             torch.nn.Dropout(0.5),
#             nn.ReLU(),
#             nn.Linear(128, 3)
#         )
#
#     def forward(self, x_heatmap, x_voxel):
#         heatmap_features = self.features_heatmap(x_heatmap)
#         voxel_features = self.features_voxels(x_voxel)
#
#         features = torch.cat((heatmap_features.view(x_heatmap.shape[0], -1), voxel_features.view(x_voxel.shape[0], -1)),
#                              dim=1)
#
#         out = self.classifier(features)
#         return out

#### New :
########### Heatmap Networks
# Used for all Resnets:
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU()]
    if pool:
      layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class model_heatmap1(nn.Module):
    def __init__(self, channels=8):
        super(model_heatmap1, self).__init__()

        self.c = channels

        self.features = nn.Sequential(
            # In : (1,32,32)
            nn.Conv2d(1, self.c, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(self.c),
            nn.ReLU(),

            # In : (c,16,16)
            nn.Conv2d(self.c, 2 * self.c, kernel_size=2, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(2 * self.c),
            nn.ReLU(),

            # In : (2c,8,8)
            nn.Conv2d(2 * self.c, self.c, kernel_size=2, stride=2, padding=0, bias=True),
            nn.ReLU(),
        )
        # Out : (batch_size,c,4,4)

    def forward(self, x):
        heat_features = self.features(x)
        return heat_features


class model_heatmap2(nn.Module):
    def __init__(self, channels=8):
        super(model_heatmap2, self).__init__()

        self.c = channels

        self.features = nn.Sequential(
            # In : (1,32,32)
            nn.Conv2d(1, self.c, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(self.c),
            nn.ReLU(),
            # In : (c,30,30)
            nn.Conv2d(self.c, 2 * self.c, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(2 * self.c),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # In : (2c,14,14)
            nn.Conv2d(2 * self.c, 2 * self.c, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(2 * self.c),
            nn.ReLU(),
            # In : (2c,12,12)
            nn.Conv2d(2 * self.c, 2 * self.c, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(2 * self.c),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # In : (c,6,6)
            nn.Conv2d(2 * self.c, self.c, kernel_size=2, stride=1, padding=0, bias=True),
            nn.ReLU(),

            # Out : (batch_size,c,4,4)
        )

    def forward(self, x):
        heat_features = self.features(x)
        return heat_features


class model_heatmapVGG(nn.Module):
    def __init__(self, channels=8):
        super(model_heatmapVGG, self).__init__()

        self.c = channels

        self.features = nn.Sequential(
            # In : (1,32,32)
            nn.Conv2d(1, self.c, 3, padding=1),  # Conv1
            nn.BatchNorm2d(self.c),
            nn.ReLU(),
            nn.Conv2d(self.c, self.c, 3, padding=1),  # Conv2
            nn.BatchNorm2d(self.c),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Pool1

            # In : (8,16,16)
            nn.Conv2d(self.c, 2 * self.c, 3, padding=1),  # Conv3
            nn.BatchNorm2d(2*self.c),
            nn.ReLU(),
            nn.Conv2d(2 * self.c, 2 * self.c, 3, padding=1),  # Conv4
            nn.BatchNorm2d(2*self.c),
            nn.ReLU(),

            # In : (8,16,16)
            nn.Conv2d(2 * self.c, 2 * self.c, 3, padding=1),  # Conv5
            nn.BatchNorm2d(2*self.c),
            nn.ReLU(),
            nn.Conv2d(2 * self.c, 4 * self.c, 3, padding=1),  # Conv6
            nn.BatchNorm2d(4*self.c),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Pool2

            # In : (16,8,8)
            nn.Conv2d(4 * self.c, 4 * self.c, 3, padding=1),  # Conv7
            nn.BatchNorm2d(4*self.c),
            nn.ReLU(),
            nn.Conv2d(4 * self.c, 2 * self.c, 3, padding=1),  # Conv8
            nn.BatchNorm2d(2*self.c),
            nn.ReLU(),

            # In : (16,8,8)
            nn.Conv2d(2 * self.c, 2 * self.c, 3, padding=1),  # Conv9
            nn.BatchNorm2d(2*self.c),
            nn.ReLU(),
            nn.Conv2d(2 * self.c, self.c, 3, padding=1),  # Conv10
            nn.BatchNorm2d(self.c),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # Pool3

            # In : (32,4,4)
            nn.Conv2d(self.c, self.c, 3, padding=1),  # Conv11
            nn.BatchNorm2d(self.c),
            nn.ReLU(),
            nn.Conv2d(self.c, self.c, 3, padding=1),  # Conv12
            nn.ReLU(),

            # Out : (c,4,4)
        )

    def forward(self, x):
        heat_features = self.features(x)
        return heat_features


# either = in -1
# or = (in -2)//2 +1

class model_heatmapResNet(nn.Module):
    def __init__(self, channels=8):
        super(model_heatmapResNet, self).__init__()

        self.c = channels

        # In : (1,32,32)
        self.conv1 = conv_block(1, 2 * self.c)
        self.bn1 = nn.BatchNorm2d(2 * self.c)
        self.conv2 = conv_block(2 * self.c, 4 * self.c, pool=True)
        self.bn2 = nn.BatchNorm2d(4 * self.c)
        self.res1 = conv_block(4 * self.c, 4 * self.c)

        self.conv3 = conv_block(4 * self.c, 4 * self.c, pool=True)
        self.bn3 = nn.BatchNorm2d(4 * self.c)
        self.res2 = conv_block(4 * self.c, 4 * self.c)

        self.conv4 = conv_block(4 * self.c, self.c, pool=True)
        self.bn4 = nn.BatchNorm2d(self.c)
        self.res3 = conv_block(self.c, self.c)

        self.ReLu = nn.ReLU()

    def forward(self, x):
        # In : (1,32,32)
        out = self.bn1(self.conv1(x))
        out = self.bn2(self.conv2(out))
        out = self.ReLu(self.res1(out) + out)

        # In : (4c,16,16)
        out = self.bn3(self.conv3(out))
        out = self.ReLu(self.res2(out) + out)

        # In : (4c,8,8)
        out = self.bn4(self.conv4(out))
        out = self.ReLu(self.res3(out) + out)

        # Out : (c,4,4)
        return out


########### Voxel Networks
class model_Voxel1(nn.Module):
    def __init__(self, channels=8):
        super(model_Voxel1, self).__init__()

        self.c = channels

        self.features_voxels = nn.Sequential(
            # In : (1,26,26,26)
            nn.Conv3d(1, 2 * self.c, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(2 * self.c),
            nn.ReLU(),
            # In : (16,24,24,24)
            nn.Conv3d(2 * self.c, 4 * self.c, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(4 * self.c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),

            # In : (32,11,11,11)
            nn.Conv3d(4 * self.c, 2 * self.c, kernel_size=3, stride=1, padding=0, bias=True),
            nn.BatchNorm3d(2 * self.c),
            nn.ReLU(),
            # In : (32,9,9,9)
            nn.Conv3d(2 * self.c, self.c, kernel_size=3, stride=1, padding=0, bias=True),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)

            # Out : (8,3,3,3)
        )

    def forward(self, x_voxel):
        voxel_features = self.features_voxels(x_voxel)

        return voxel_features


class model_VoxelVGG(nn.Module):
    def __init__(self, channels=8):
        super(model_VoxelVGG, self).__init__()
        self.c = channels
        self.features_voxels = nn.Sequential(
            # In : (1,26,26,26)
            nn.Conv3d(1, self.c, 3, padding=1),  # Conv1
            nn.BatchNorm3d(self.c),
            nn.ReLU(),
            nn.Conv3d(self.c, self.c, 3, padding=1),  # Conv2
            nn.BatchNorm3d(self.c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Pool1

            # In : (8,13,13,13)
            nn.Conv3d(self.c, 2 * self.c, 3, padding=1),  # Conv3
            nn.BatchNorm3d(2 * self.c),
            nn.ReLU(),
            nn.Conv3d(2 * self.c, 2 * self.c, 3, padding=1),  # Conv4
            nn.BatchNorm3d(2 * self.c),
            nn.ReLU(),

            # In : (8,13,13,13)
            nn.Conv3d(2 * self.c, 2 * self.c, 3, padding=1),  # Conv5
            nn.BatchNorm3d(2 * self.c),
            nn.ReLU(),
            nn.Conv3d(2 * self.c, 4 * self.c, 3, padding=1),  # Conv6
            nn.BatchNorm3d(4 * self.c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Pool2

            # In : (8,6,6,6)
            nn.Conv3d(4 * self.c, 4 * self.c, 3, padding=1),  # Conv7
            nn.BatchNorm3d(4 * self.c),
            nn.ReLU(),
            nn.Conv3d(4 * self.c, 2 * self.c, 3, padding=1),  # Conv8
            nn.BatchNorm3d(2 * self.c),
            nn.ReLU(),

            # In : (8,6,6,6)
            nn.Conv3d(2 * self.c, 2 * self.c, 3, padding=1),  # Conv9
            nn.BatchNorm3d(2 * self.c),
            nn.ReLU(),
            nn.Conv3d(2 * self.c, self.c, 3, padding=1),  # Conv10
            nn.BatchNorm3d(self.c),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # Pool3

            # In : (8,3,3,3)
            nn.Conv3d(self.c, self.c, 3, padding=1),  # Conv11
            nn.BatchNorm3d(self.c),
            nn.ReLU(),
            nn.Conv3d(self.c, self.c, 3, padding=1),  # Conv12
            nn.ReLU(),

            # Out : : (8,3,3,3)
        )

    def forward(self, x_voxel):
        voxel_features = self.features_voxels(x_voxel)
        return voxel_features


#### Combining :
class model_heat_vox(nn.Module):
    def __init__(self, heat_model=1, vox_model=1, batchNorm=True, dropout=0.5, channels=8):
        super(model_heat_vox, self).__init__()

        self.heat_model = heat_model  # 4 poss
        self.vox_model = vox_model  # 2 poss
        self.dropout = dropout  # 3 poss
        self.c = channels

        if heat_model == "1":
            self.features_heatmap = model_heatmap1(channels=self.c)
        elif heat_model == "2":
            self.features_heatmap = model_heatmap2(channels=self.c)
        elif heat_model == "VGG":
            self.features_heatmap = model_heatmapVGG(channels=self.c)
        elif heat_model == "ResNet":
            self.features_heatmap = model_heatmapResNet(channels=self.c)

        if vox_model == "1":
            self.features_voxels = model_Voxel1(channels=self.c)
        elif vox_model == "VGG":
            self.features_voxels = model_VoxelVGG(channels=self.c)

        # Classfier : (not defined in sequential to make sure dropout is off when model.eval()
        self.drop_layer = nn.Dropout(p=dropout)

        self.fc1 = nn.Linear(channels * (4 * 4 + 3 * 3 * 3), 128)
        self.fc2 = nn.Linear(128, 1)

        # self.classifier = nn.Sequential(
        #     nn.Linear(2*2*32 + 8*3*3*3, 128), # 344
        #     torch.nn.Dropout(0.5),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )

    def forward(self, x_heatmap, x_voxel):
        heatmap_features = self.features_heatmap(x_heatmap)
        voxel_features = self.features_voxels(x_voxel)

        features = torch.cat((heatmap_features.view(x_heatmap.shape[0], -1), voxel_features.view(x_voxel.shape[0], -1)),
                             dim=1)

        out = self.fc1(features)
        out = self.drop_layer(out)
        out = nn.ReLU()(out)
        out = self.fc2(out)
        # out = self.classifier(features)
        return out


class Normalize_Voxel(object):
    def __init__(self, mean=0, std=1):
        self.mean = mean
        self.std = std  # Note: flipping alond dim 0 will not result in a change, use with 1,2 or 3

    def __call__(self, tensor):
        return (tensor - self.mean) / self.std

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0},std={1})'.format(self.mean, self.std)