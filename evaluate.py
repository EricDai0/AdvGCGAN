import torch
import torch.nn as nn
import torch.optim as optim

from data.dataset_benchmark import BenchmarkDataset
from model.gan_network_sn import Generator, Discriminator
from model.gradient_penalty import GradientPenalty

from models.pointnet import PointNetCls, feature_transform_regularizer
from models.pointnet2 import PointNet2ClsMsg
from models.dgcnn import DGCNN
from models.pointcnn import PointCNNCls
from torch.autograd import Variable
from evaluation import *

from arguments import Arguments
from defense import SRS, SOR

import time
import visdom
import numpy as np
import os
import h5py


os.environ['CUDA_VISIBLE_DEVICES'] = '4'


def get_average_dis(x, k=20, num=-1):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]
        idx
    """
    
    x = x.permute(0,2,1)
    B, dims, N = x.shape

    # ----------------------------------------------------------------
    # batched pair-wise distance in feature space maybe is can be changed to coordinate space
    # ----------------------------------------------------------------
    xt = x.permute(0, 2, 1)
    xi = -2 * torch.bmm(xt, x)
    xs = torch.sum(xt**2, dim=2, keepdim=True)
    xst = xs.permute(0, 2, 1)
    dist = xi + xs + xst # [B, N, N]
    
    k_dist = dist[:,:,1:k+1]
    std_dist = k_dist.std(dim = 2)
    std_mean = torch.mean(std_dist, 1, True)
    std_dist = std_dist - std_mean
    return torch.mean(std_dist)
    
class TreeGAN():
    def __init__(self, args):
        self.args = args
        # ------------------------------------------------Dataset---------------------------------------------- #
        self.data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, uniform=None, class_choice=args.class_choice)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        print("Training Dataset : {} prepared.".format(len(self.data)))
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        self.G = Generator(batch_size=args.batch_size, features=args.G_FEAT, degrees=args.DEGREE, support=args.support).to(args.device)
        self.D = Discriminator(batch_size=args.batch_size, features=args.D_FEAT, classes=3).to(args.device)             
        
        self.optimizerG = optim.Adam(self.G.parameters(), lr=args.lr, betas=(0, 0.99))
        self.optimizerD = optim.Adam(self.D.parameters(), lr=args.lr, betas=(0, 0.99))

        self.GP = GradientPenalty(args.lambdaGP, gamma=1, device=args.device)
        print("Network prepared.")
        # ----------------------------------------------------------------------------------------------------- #

        # ---------------------------------------------Visualization------------------------------------------- #
        self.vis = visdom.Visdom(port=args.visdom_port)
        assert self.vis.check_connection()
        print("Visdom connected.")
        # ----------------------------------------------------------------------------------------------------- #

    def run(self, save_ckpt=None, load_ckpt=None, result_path=None):        
        color_num = self.args.visdom_color
        chunk_size = int(self.args.point_num / color_num)
        colors = np.array([(227,0,27),(231,64,28),(237,120,15),(246,176,44),
                           (252,234,0),(224,221,128),(142,188,40),(18,126,68),
                           (63,174,0),(113,169,156),(164,194,184),(51,186,216),
                           (0,152,206),(16,68,151),(57,64,139),(96,72,132),
                           (172,113,161),(202,174,199),(145,35,132),(201,47,133),
                           (229,0,123),(225,106,112),(163,38,42),(128,128,128)])
        colors = colors[np.random.choice(len(colors), color_num, replace=False)]
        label_visdom = torch.stack([torch.ones(chunk_size).type(torch.LongTensor) * inx for inx in range(1,int(color_num)+1)], dim=0).view(-1)

        epoch_log = 0
        
        loss_log = {'G_loss': [], 'D_loss': []}
        loss_legend = list(loss_log.keys())
        closs = nn.CrossEntropyLoss()
        softmax_func=nn.Softmax()
        
        
        num_classes = 16
        checkpoint = torch.load('checkpoints/%s.pth' % args.model_path)
        if self.args.model == 'pointnet':
            model = PointNetCls(num_classes, 1)  
            model = model.to(self.args.device)  
        elif self.args.model == 'pointnet2':
            model = PointNet2ClsMsg(num_classes)
            model = model.to(self.args.device)
            model = nn.DataParallel(model)
        elif self.args.model == 'dgcnn':
            model = DGCNN(num_classes)
            model = model.to(self.args.device) 
            model = nn.DataParallel(model)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print('Successfully loaded!')
        print('Stored test accuracy %.3f%%' % checkpoint['acc_list'][-1])
        #print(self.D)
        metric = {'FPD': []}
        if load_ckpt is not None:
            checkpoint = torch.load(load_ckpt, map_location=self.args.device)
            self.D.load_state_dict(checkpoint['D_state_dict'])
            self.G.load_state_dict(checkpoint['G_state_dict'])

            epoch_log = checkpoint['epoch']

            loss_log['G_loss'] = checkpoint['G_loss']
            loss_log['D_loss'] = checkpoint['D_loss']
            loss_legend = list(loss_log.keys())

            metric['FPD'] = checkpoint['FPD']
            
            print("Checkpoint loaded.")
        
        correct = 0
        total = 0
        correct_sor = 0
        correct_srs = 0
        
        for _iter, data in enumerate(self.dataLoader):

            start_time = time.time()
            point, label = data
            point = point.to(self.args.device)
            label = label.to(self.args.device)

            label = torch.squeeze(label)
            label = Variable(torch.from_numpy(np.ones(label.shape[0],).astype(np.int64))).cuda().long()*self.args.target_class
            
            z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
            z_label = label.unsqueeze(1).unsqueeze(2)
            if label.shape[0]<self.args.batch_size:
                time_r = self.args.batch_size/label.shape[0]
                z_label = z_label.repeat(int(time_r)+1,1 ,1)
            z_label = z_label[:self.args.batch_size,:,:]
            z_label = z_label.expand(self.args.batch_size, 1, 96)
            z = torch.cat([z, z_label], dim=2)
            tree = [z]
            
            with torch.no_grad():
                fake_point = self.G(tree)  
                G_fake, G_label = self.D(fake_point)
            
            fake_point = fake_point.permute(0, 2, 1)
            
            #fake_point = point.permute(0, 2, 1)
            target_label, _ = model(fake_point)
                
            
            fake_point = fake_point.permute(0, 2, 1)
            true_label = Variable(torch.from_numpy(np.ones(label.shape[0],).astype(np.int64))).cuda().long()
            tar_label = Variable(torch.from_numpy(np.ones(label.shape[0],).astype(np.int64))).cuda().long()
            
            
            total += tar_label.shape[0]
            
            for i in range(label.shape[0]):
                if label[i] == self.args.target_class: 
                    true_label[i] = self.args.target_true_class;       
                
                            
            rates, indices = target_label.sort(1, descending=True) 
            rates, indices = rates.squeeze(0), indices.squeeze(0)    
            #print(indices)
            adv_loss = 0
            for i in range(true_label.shape[0]):
                if true_label[i] == indices[i][0]:  # classify is correct
                    tar_label[i] = indices[i][1]
                else:
                    correct += 1
                    tar_label[i] = -1
            #print(indices)            
            if _iter == 0:
                orign_data = point
                fake_data = fake_point
            else:
                orign_data = torch.cat((orign_data, point),0)
                fake_data = torch.cat((fake_data, fake_point),0)     
           
            fake_srs = SRS(fake_point.cpu().numpy())
            fake_sor = SOR(fake_point.cpu().numpy())
            
            fake_srs = torch.from_numpy(fake_srs).cuda().float()
            fake_sor = torch.from_numpy(fake_sor).cuda().float()
            
            fake_srs = fake_srs.permute(0, 2, 1)
            fake_sor = fake_sor.permute(0, 2, 1)
            
            srs_label, _ = model(fake_srs)
            sor_label, _ = model(fake_sor)
            
            rates, indices = srs_label.sort(1, descending=True) 
            rates, indices = rates.squeeze(0), indices.squeeze(0)  
            
            for i in range(true_label.shape[0]):
                if true_label[i] == indices[i][0]:  # classify is correct
                    tar_label[i] = indices[i][1]
                else:
                    correct_srs += 1
                    tar_label[i] = -1
                    
            rates, indices = sor_label.sort(1, descending=True) 
            rates, indices = rates.squeeze(0), indices.squeeze(0)  
            
            for i in range(true_label.shape[0]):
                if true_label[i] == indices[i][0]:  # classify is correct
                    tar_label[i] = indices[i][1]
                else:
                    correct_sor += 1
                    tar_label[i] = -1
                    
            if _iter == 0:
                srs_data = fake_srs
                sor_data = fake_sor
            else:
                srs_data = torch.cat((srs_data, fake_srs),0)
                sor_data = torch.cat((sor_data, fake_sor),0)  
        
        print('Acc: %f' % correct/total * 100)            
        print('SRS Acc: %f' % correct_srs/total * 100)            
        print('SOR Acc: %f' % correct_sor/total * 100)
        
        np.save("fake_data.npy",fake_data.cpu())
        np.save("orign_data.npy",orign_data.cpu())
        asd
        print(fake_data.shape)
        print(orign_data.shape)
        current_dir = 'data.h5'
        h5f = h5py.File(current_dir, 'w')
        h5f.create_dataset('orig_data', data=orign_data.cpu())
        h5f.create_dataset('data', data=fake_data.cpu())
        h5f.create_dataset('orig_label', data=label.cpu()*2)
        h5f.create_dataset('label', data=label.cpu()*2)
        h5f.close()
        f = open("log-table-adv.txt", "w")  
            
        print(correct/total * 100, file = f)            
        print(correct_srs/total * 100, file = f)            
        print(correct_sor/total * 100, file = f)
        print('SAVED!')
        with torch.no_grad():
            results = compute_all_metrics(fake_data.to(self.args.device), orign_data.to(self.args.device), 64 , accelerated_cd=True)
            results = {k:v.item() for k, v in results.items()}
            jsd = jsd_between_point_cloud_sets(fake_data.cpu().numpy(), orign_data.cpu().numpy())
            results['jsd'] = jsd 
            print('JSD: %f' % jsd)
            for k,v in results.items():
                print(k,v, file = f)
        f.flush()
    
        f.close()

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    SAVE_CHECKPOINT = args.ckpt_path + args.ckpt_save if args.ckpt_save is not None else None
    LOAD_CHECKPOINT = args.ckpt_path + args.ckpt_load if args.ckpt_load is not None else None
    RESULT_PATH = args.result_path + args.result_save

    GANmodel = TreeGAN(args)
    GANmodel.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT, result_path=RESULT_PATH)
