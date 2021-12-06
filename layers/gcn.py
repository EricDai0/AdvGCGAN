import torch
import torch.nn as nn
import torch.nn.init as init
import math

class TreeGCN(nn.Module):
    def __init__(self, batch, depth, features, degrees, support=10, node=1, upsample=False, activation=True):
        self.batch = batch
        self.depth = depth
        self.in_feature = features[depth]
        self.out_feature = features[depth+1]
        self.node = node
        self.degree = degrees[depth]
        self.upsample = upsample
        self.activation = activation
        super(TreeGCN, self).__init__()

        self.W_root = nn.ModuleList([nn.Linear(features[inx], self.out_feature, bias=False) for inx in range(self.depth+1)])

        if self.upsample:
            self.W_branch = nn.Parameter(torch.FloatTensor(self.node, self.in_feature, self.degree*self.in_feature))

        if self.node > 1024:
            self.N_L = nn.Sequential(
                        nn.Conv2d(self.in_feature*2, self.in_feature*4, [1, 20//2+1], [1, 1]),  # Fin, Fout, kernel_size, stride
                        nn.BatchNorm2d(self.in_feature*4),
                        nn.LeakyReLU(inplace=True)
                    )
            self.N_branch = nn.Sequential(nn.Conv2d(self.node, self.node, [1, 20], [1, 1]),
                                    nn.BatchNorm2d(self.node))
            self.N_fea = nn.Linear(self.in_feature*2, self.in_feature)
        
        self.W_loop = nn.Sequential(nn.Linear(self.in_feature, self.in_feature*support, bias=False),
                                    nn.Linear(self.in_feature*support, self.out_feature, bias=False))

        self.bias = nn.Parameter(torch.FloatTensor(1, self.degree, self.out_feature))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.init_param()

    def init_param(self):
        if self.upsample:
            init.xavier_uniform_(self.W_branch.data, gain=init.calculate_gain('relu'))
            
        stdv = 1. / math.sqrt(self.out_feature)
        self.bias.data.uniform_(-stdv, stdv)
        
    def get_edge_features_xyz(self, x, k=20, num=-1):
        """
        Args:
            x: point cloud [B, dims, N]
            k: kNN neighbours
        Return:
            [B, 2dims, N, k]
            idx
        """
        B, dims, N = x.shape

        # ----------------------------------------------------------------
        # batched pair-wise distance in feature space maybe is can be changed to coordinate space
        # ----------------------------------------------------------------
        xt = x.permute(0, 2, 1)
        xi = -2 * torch.bmm(xt, x)
        xs = torch.sum(xt**2, dim=2, keepdim=True)
        xst = xs.permute(0, 2, 1)
        dist = xi + xs + xst # [B, N, N]
        
        # get k NN id    
        _, idx_o = torch.sort(dist, dim=2)
        idx = idx_o[: ,: ,1:k+1] # [B, N, k]
        idx = idx.contiguous().view(B, N*k)


        # gather
        neighbors = []
        for b in range(B):
            tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
            tmp = tmp.view(dims, N, k)
            neighbors.append(tmp)

        neighbors = torch.stack(neighbors)  # [B, d, N, k]

        # centralize
        central = x.unsqueeze(3).repeat(1, 1, 1, k)         # [B, d, N, 1] -> [B, d, N, k]

        e_fea = neighbors - central
        
        
        e_fea = torch.cat((central, e_fea), 1)  
        
        
        return e_fea

    def forward(self, tree):
        
        root = 0
        for inx in range(self.depth+1):
            root_num = tree[inx].size(1)
            repeat_num = int(self.node / root_num)
            root_node = self.W_root[inx](tree[inx])
            root = root + root_node.repeat(1,1,repeat_num).view(self.batch,-1,self.out_feature)

        branch = 0
        if self.upsample and self.node <= 1024:
            branch = tree[-1].unsqueeze(2) @ self.W_branch
            branch = self.leaky_relu(branch)
            branch = branch.view(self.batch,self.node*self.degree,self.in_feature)
            
            branch = self.W_loop(branch)

            branch = root.repeat(1,1,self.degree).view(self.batch,-1,self.out_feature) + branch
        else:
            if self.node > 1024:
                points = tree[-1].permute(0, 2, 1)
                branch = self.get_edge_features_xyz(points)
                
                B, C, N, K = branch.size()
                branch = self.N_L(branch)
                branch = branch.transpose(2, 1)                                 # BxNx2Cxk/2
                branch = branch.contiguous().view(B, N, C, 2, K//2)         # BxNxCx2x(k//2+1)
                branch = branch.contiguous().view(B, N, C, K)               # BxNxCx(k+2)
                
                branch = self.N_branch(branch)         
                branch = branch.view(self.batch,self.node,self.in_feature*2)
                
                points = tree[-1].unsqueeze(2) @ self.W_branch
                points = self.leaky_relu(points)
                points = points.view(self.batch,self.node,self.in_feature*2)
                
                branch = torch.cat((points, branch), 1)  
                
                branch = self.N_fea(branch)
                
                branch = self.leaky_relu(branch)
                
                branch = branch.view(self.batch,self.node*self.degree,self.in_feature)

                branch = self.W_loop(branch)

                branch = root.repeat(1,1,self.degree).view(self.batch,-1,self.out_feature) + branch
        if not self.upsample:
            branch = self.W_loop(tree[-1])

            branch = root + branch

        if self.activation:
            branch = self.leaky_relu(branch + self.bias.repeat(1,self.node,1))
            
            
        tree.append(branch)
        return tree