import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from layers.gcn import TreeGCN

from math import ceil


class ResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        
        self.model = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.utils.spectral_norm(self.conv1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.utils.spectral_norm(self.conv2)
            )
        self.bypass = nn.Sequential()
        self.bypass_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
        self.bypass = nn.Sequential(
            nn.utils.spectral_norm(self.bypass_conv)
        )


    def forward(self, x):
        return self.model(x) + self.bypass(x)
    
class FirstResBlockDiscriminator(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super(FirstResBlockDiscriminator, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, stride=1)
        nn.init.xavier_uniform(self.conv1.weight.data, 1.)
        nn.init.xavier_uniform(self.conv2.weight.data, 1.)
        
        self.model = nn.Sequential(
            nn.utils.spectral_norm(self.conv1),
            nn.LeakyReLU(negative_slope=0.2),
            nn.utils.spectral_norm(self.conv2)
            )
        self.bypass = nn.Sequential()
        self.bypass_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1)
        nn.init.xavier_uniform(self.bypass_conv.weight.data, np.sqrt(2))
        self.bypass = nn.Sequential(
            nn.utils.spectral_norm(self.bypass_conv)
        )


    def forward(self, x):
        return self.model(x) + self.bypass(x)


class Discriminator(nn.Module):
    def __init__(self, batch_size, features, classes):
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        super(Discriminator, self).__init__()

        self.fc_layers = nn.ModuleList([])
        for inx in range(self.layer_num):
            if inx == 0:
                self.fc_layer = FirstResBlockDiscriminator(features[inx], features[inx+1], stride=1)
                self.fc_layers.append(self.fc_layer)
            else:
                self.fc_layer = ResBlockDiscriminator(features[inx], features[inx+1], stride=1)
                self.fc_layers.append(self.fc_layer)
        
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.l1 = nn.Linear(features[-1], features[-2])
        self.l2 = nn.Linear(features[-2], 1)
        nn.init.xavier_uniform(self.l1.weight.data, 1.)
        nn.init.xavier_uniform(self.l2.weight.data, 1.)
        self.l1 = nn.utils.spectral_norm(self.l1)
        self.l2 = nn.utils.spectral_norm(self.l2)
        
        self.final_layer = nn.Sequential(self.l1,
                                         self.l2)
        self.l3 = nn.Linear(features[-1], features[-2])
        self.l4 = nn.Linear(features[-2], classes)
        nn.init.xavier_uniform(self.l3.weight.data, 1.)
        nn.init.xavier_uniform(self.l4.weight.data, 1.)
        
        self.cls_layer = nn.Sequential(self.l3,
                                         self.l4)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, f):
        feat = f.transpose(1,2)
        vertex_num = feat.size(2)

        for inx in range(self.layer_num):
            feat = self.fc_layers[inx](feat)
        
        feat = self.leaky_relu(feat)
        fea = F.max_pool1d(input=feat, kernel_size=vertex_num).squeeze(-1)
        out = self.final_layer(fea) # (B, 1)
        label = self.cls_layer(fea) # (B, 1)
        label = self.softmax(label)
        
        return out, label


class Generator(nn.Module):
    def __init__(self, batch_size, features, degrees, support):
        self.batch_size = batch_size
        self.layer_num = len(features)-1
        assert self.layer_num == len(degrees), "Number of features should be one more than number of degrees."
        self.pointcloud = None
        super(Generator, self).__init__()
        
        vertex_num = 1
        self.gcn = nn.Sequential()
        for inx in range(self.layer_num):
            if inx == self.layer_num-1:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=False))
            else:
                self.gcn.add_module('TreeGCN_'+str(inx),
                                    TreeGCN(self.batch_size, inx, features, degrees, 
                                            support=support, node=vertex_num, upsample=True, activation=True))
            vertex_num = int(vertex_num * degrees[inx])

    def forward(self, tree):
        
        feat = self.gcn(tree)
        
        self.pointcloud = feat[-1]

        return self.pointcloud

    def getPointcloud(self):
        return self.pointcloud[-1]