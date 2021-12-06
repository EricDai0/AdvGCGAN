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

from arguments import Arguments

import time
import visdom
import numpy as np
import os,sys

import math
import pointnet2_ops.pointnet2_utils as pn2_utils
from knn_cuda import KNN

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

def uniform_loss(pcd,knn_uniform, percentage=[0.004,0.006,0.008,0.010,0.012],radius=1.0):
    pcd = pcd.permute(0, 2, 1)
    B,N,C=pcd.shape[0],pcd.shape[1],pcd.shape[2]
    npoint=int(N*0.05)
    loss=0
    further_point_idx = pn2_utils.furthest_point_sample(pcd.permute(0, 2, 1).contiguous(), npoint)

    new_xyz = pn2_utils.gather_operation(pcd.permute(0, 2, 1).contiguous(), further_point_idx)  # B,C,N

    for p in percentage:
        nsample=int(N*p)
        r=math.sqrt(p*radius)
        disk_area=math.pi*(radius**2)/N

        idx=pn2_utils.ball_query(r,nsample,pcd.contiguous(),new_xyz.permute(0,2,1).contiguous()) #b N nsample

        expect_len=math.sqrt(disk_area)

        grouped_pcd=pn2_utils.grouping_operation(pcd.permute(0,2,1).contiguous(),idx)#B C N nsample
        grouped_pcd=grouped_pcd.permute(0,2,3,1) #B N nsample C
        
        grouped_pcd=torch.cat(torch.unbind(grouped_pcd,dim=1),dim=0)#B*N nsample C
        
        dist,_=knn_uniform(grouped_pcd,grouped_pcd)
        #print(dist.shape)
        uniform_dist=dist[:,:,1:] #B*N nsample 1
        uniform_dist=torch.abs(uniform_dist+1e-8)
        uniform_dist=torch.mean(uniform_dist,dim=1)
        uniform_dist=(uniform_dist-expect_len)**2/(expect_len+1e-8)
        mean_loss=torch.mean(uniform_dist)
        mean_loss=mean_loss*math.pow(p*100,2)
        loss+=mean_loss
    return loss/len(percentage)
    
class AdvGCGAN():
    def __init__(self, args):
        self.args = args
        # ------------------------------------------------Dataset---------------------------------------------- #
        self.data = BenchmarkDataset(root=args.dataset_path, npoints=args.point_num, uniform=None, class_choice=args.class_choice)
        self.dataLoader = torch.utils.data.DataLoader(self.data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        print("Training Dataset : {} prepared.".format(len(self.data)))
        # ----------------------------------------------------------------------------------------------------- #

        # -------------------------------------------------Module---------------------------------------------- #
        self.G = Generator(batch_size=args.batch_size, features=args.G_FEAT, degrees=args.DEGREE, support=args.support).to(args.device)
        self.D = Discriminator(batch_size=args.batch_size, features=args.D_FEAT, classes=args.num_class).to(args.device)             
        
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
        closs = nn.NLLLoss()
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
        
        model = model.to(self.args.device)
        model.eval()
        print('Successfully loaded!')
        print('Stored test accuracy %.3f%%' % checkpoint['acc_list'][-1])
        
        
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
        
        knn_uniform=KNN(k=2,transpose_mode=True)
        
        for epoch in range(epoch_log, self.args.epochs):
            correct = 0
            total = 0
            for _iter, data in enumerate(self.dataLoader):
                # Start Time
                start_time = time.time()
                point, label = data
                point = point.to(self.args.device)
                label = label.to(self.args.device)

                label = torch.squeeze(label)
                # -------------------- Discriminator -------------------- #
                for d_iter in range(self.args.D_iter):
                    self.D.zero_grad()
                    
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
                    
                    
                    D_real, real_label = self.D(point)
                    D_realm = D_real.mean()
                    
                    real_cls = closs(real_label, label)
                    d_loss = -D_realm + 5 * real_cls
                    
                    d_loss.backward()
                    
                    fake_point = self.G(tree) 
                    D_fake, fake_label = self.D(fake_point)
                    D_fakem = D_fake.mean()
                    fake_cls = closs(fake_label[:label.shape[0]], label)
                    
                    d_fake_loss = D_fakem + 5 * fake_cls
                    d_fake_loss.backward()
                    
                    gp_loss = self.GP(self.D, point.data, fake_point.data)
                    gp_loss.backward()
                    
                    self.optimizerD.step()
                    
                loss_log['D_loss'].append(d_loss.item())                  
                
                
                # ---------------------- Generator ---------------------- #
                self.G.zero_grad()
                
                z = torch.randn(self.args.batch_size, 1, 96).to(self.args.device)
                z_label = label.unsqueeze(1).unsqueeze(2)
                if label.shape[0]<self.args.batch_size:
                    time_r = self.args.batch_size/label.shape[0]
                    z_label = z_label.repeat(int(time_r)+1,1 ,1)
                z_label = z_label[:self.args.batch_size,:,:]
                z_label = z_label.expand(self.args.batch_size, 1, 96)
                z = torch.cat([z, z_label], dim=2)
                tree = [z]
                
                fake_point = self.G(tree)
                G_fake, G_label = self.D(fake_point)
                G_fakem = G_fake.mean()
                G_cls = closs(G_label[:label.shape[0]], label)
                
                fake_point = fake_point.permute(0, 2, 1)
            # ---------------------- Adversarial Attack --------------------- #                
                str_loss = uniform_loss(fake_point, knn_uniform)
                target_label, _ = model(fake_point)
                dis_label = torch.argmax(G_label, dim=1)
                true_label = Variable(torch.from_numpy(np.ones(label.shape[0],).astype(np.int64))).cuda().long()
                tar_label = Variable(torch.from_numpy(np.ones(label.shape[0],).astype(np.int64))).cuda().long()
                
                total += tar_label.shape[0]
                
                for i in range(label.shape[0]):
                    if label[i] == self.args.target_class: 
                        true_label[i] = self.args.target_true_class;       
                
                rates, indices = target_label.sort(1, descending=True) 
                rates, indices = rates.squeeze(0), indices.squeeze(0)    
                
                adv_loss = 0
                
                if self.args.target_attack_label is not None:
                    for i in range(true_label.shape[0]):
                        if true_label[i] != self.args.target_attack_label:  # classify is correct
                            tar_label[i] = self.args.target_attack_label
                        else:
                            correct += 1
                            tar_label[i] = -1
                else:                     
                    for i in range(true_label.shape[0]):
                        if true_label[i] == indices[i][0]:  # classify is correct
                            tar_label[i] = indices[i][1]
                        else:
                            correct += 1
                            tar_label[i] = -1
                        
                target_logits = (tar_label != -1).nonzero()
                target_logits = target_logits.squeeze()
                
                
                target_log = target_label[target_logits]
                tar_label = tar_label[target_logits]
                if target_logits.dim() == 0:
                    target_log = target_log.unsqueeze(0)
                    tar_label = tar_label.unsqueeze(0)
                    
                
                if target_logits.shape != torch.Size([0]):
                    adv_loss = closs(target_log, tar_label)
               
                g_loss = -G_fakem + 5 * G_cls + str_loss + 0.05 * adv_loss
                g_loss.backward()
                self.optimizerG.step()
                loss_log['G_loss'].append(g_loss.item())

                # --------------------- Visualization -------------------- #

                print("[Epoch/Iter] ", "{:3} / {:3}".format(epoch, _iter),
                      "[ D_Loss ] ", "{: 7.6f}".format(d_loss), 
                      "[ G_Loss ] ", "{: 7.6f}".format(g_loss), 
                      "[ Time ] ", "{:4.2f}s".format(time.time()-start_time),
                      "[ Acc ]", "{:4.2f}%".format(correct/total * 100))

                
                if _iter % 10 == 0:
                    generated_point = self.G.getPointcloud()
                    plot_X = np.stack([np.arange(len(loss_log[legend])) for legend in loss_legend], 1)
                    plot_Y = np.stack([np.array(loss_log[legend]) for legend in loss_legend], 1)

                    self.vis.line(X=plot_X, Y=plot_Y, win=1,
                                  opts={'title': 'AdvGCGAN Loss', 'legend': loss_legend, 'xlabel': 'Iteration', 'ylabel': 'Loss'})

                    self.vis.scatter(X=generated_point[:,torch.LongTensor([2,0,1])], Y=label_visdom, win=2,
                                     opts={'title': "Generated Pointcloud", 'markersize': 2, 'markercolor': colors, 'webgl': True})

                    if len(metric['FPD']) > 0:
                        self.vis.line(X=np.arange(len(metric['FPD'])), Y=np.array(metric['FPD']), win=3, 
                                      opts={'title': "Frechet Pointcloud Distance", 'legend': ["{} / FPD best : {:.6f}".format(np.argmin(metric['FPD']), np.min(metric['FPD']))]})

                    print('Figures are saved.')
                    
            # ---------------------- Save checkpoint --------------------- #
            if epoch % 4 == 0 and not save_ckpt == None:
                torch.save({
                        'epoch': epoch,
                        'D_state_dict': self.D.state_dict(),
                        'G_state_dict': self.G.state_dict(),
                        'D_loss': loss_log['D_loss'],
                        'G_loss': loss_log['G_loss'],
                        'FPD': metric['FPD']
                }, save_ckpt+str(epoch)+ self.args.save_suffix +'.pt')

                print('Checkpoint is saved.')
            
                
                    

if __name__ == '__main__':
    args = Arguments().parser().parse_args()

    args.device = torch.device('cuda:'+str(args.gpu) if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(args.device)

    SAVE_CHECKPOINT = args.ckpt_path + args.ckpt_save if args.ckpt_save is not None else None
    LOAD_CHECKPOINT = args.ckpt_path + args.ckpt_load if args.ckpt_load is not None else None
    RESULT_PATH = args.result_path + args.result_save

    GANmodel = AdvGCGAN(args)
    GANmodel.run(save_ckpt=SAVE_CHECKPOINT, load_ckpt=LOAD_CHECKPOINT, result_path=RESULT_PATH)
