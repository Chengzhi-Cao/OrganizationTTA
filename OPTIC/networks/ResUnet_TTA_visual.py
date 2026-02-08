from torch import nn
import torch
from networks.resnet_TTA import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F
from utils.convert import *
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import random
import os
from PIL import Image

class SaveFeatures():
    def __init__(self, m, n):
        self.features = None
        self.name = n
        self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)
        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        out = self.bn(F.relu(cat_p))
        return out


class ResUnet(nn.Module):
    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False, convert=True, newBN=AdaBN, warm_n=5,save_path = None):
        super().__init__()
        if resnet == 'resnet34':
            base_model = resnet34
            bottleneck = False
            feature_channels = [64, 64, 128, 256, 512]
        elif resnet == 'resnet50':
            base_model = resnet50
            bottleneck = True
            feature_channels = [64, 256, 512, 1024, 2048]
        else:
            raise Exception('The Resnet Model only accept resnet34 and resnet50!')

        self.res = base_model(pretrained=pretrained)
        self.global_num = 0
        self.num_classes = num_classes
        self.save_path = save_path

        self.up1 = UnetBlock(feature_channels[4], feature_channels[3], 256)
        self.up2 = UnetBlock(256, feature_channels[2], 256)
        self.up3 = UnetBlock(256, feature_channels[1], 256)
        self.up4 = UnetBlock(256, feature_channels[0], 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.seg_head = nn.Conv2d(32, self.num_classes, 1)

        # Convert BN layer
        self.newBN = newBN
        if convert:
            self.res = convert_encoder_to_target(self.res, newBN, start=0, end=5, verbose=False, bottleneck=bottleneck, warm_n=warm_n)
            self.up1, self.up2, self.up3, self.up4, self.bnout = convert_decoder_to_target(
                [self.up1, self.up2, self.up3, self.up4, self.bnout], newBN, start=0, end=5, verbose=False, warm_n=warm_n)

        # Save the output feature of each BN layer.
        self.feature_hooks = []
        layers = [self.res.bn1, self.res.layer1, self.res.layer2, self.res.layer3, self.res.layer4]
        for i, layer in enumerate(layers):
            if i == 0:
                self.feature_hooks.append(SaveFeatures(layer, 'first_bn'))
            else:
                for j, block in enumerate(layer):
                    self.feature_hooks.append(SaveFeatures(block.bn1, str(i)+'-bn1'))      # BasicBlock
                    self.feature_hooks.append(SaveFeatures(block.bn2, str(i)+'-bn2'))      # BasicBlock
                    if resnet == 'resnet50':
                        self.feature_hooks.append(SaveFeatures(block.bn3, str(i)+'-bn3'))  # Bottleneck
                    if block.downsample is not None:
                        self.feature_hooks.append(SaveFeatures(block.downsample[1], str(i)+'-downsample_bn'))
        self.feature_hooks.append(SaveFeatures(self.up1.bn, '1-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.up2.bn, '2-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.up3.bn, '3-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.up4.bn, '4-up_bn'))
        self.feature_hooks.append(SaveFeatures(self.bnout, 'last_bn'))

    def change_BN_status(self, new_sample=True):
        for nm, m in self.named_modules():
            if isinstance(m, self.newBN):
                m.new_sample = new_sample

    def reset_sample_num(self):
        for nm, m in self.named_modules():
            if isinstance(m, self.newBN):
                m.new_sample = 0

    def forward(self, x):       # x=[1,3,512,512]
        
        
        # ######################################################
        # _save_path = os.path.join(self.save_path,'x')
        # if not os.path.exists(_save_path):
        #     os.makedirs(_save_path)
        # _x = x[0,:,:,:].cpu().data.numpy() 
        # _x = 255*(_x - np.min(_x)) /(np.max(_x)-np.min(_x))
        # _x =  np.uint8(_x)
        # _x = _x.transpose(1,2,0)
        # _x = Image.fromarray(_x)
        # img_x_path = '{}_x.jpg'.format(self.global_num)
        # _x.save(os.path.join(_save_path,img_x_path))
        
        
        x, sfs = self.res(x)    # x=[1,512,16,16]
        
        
        # _x = x[0,:,:,:].cpu().data.numpy() 
        # _x = np.resize(_x,(512,16*16))
        
        
        x = F.relu(x)

        x = self.up1(x, sfs[3])# x=[1,256,32,32]
        x = self.up2(x, sfs[2])# x=[1,256,64,64]


        # # ######################################################
        # _save_path = os.path.join(self.save_path,'up2')
        # if not os.path.exists(_save_path):
        #     os.makedirs(_save_path)
            
        # _fea = x[0,:,:,:].cpu().data.numpy()
        # _fea_mean = np.mean(_fea, axis=0)
        # _a = np.clip(_fea_mean, 0, 1) # 将numpy数组约束在[0, 1]范围内
        # trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        # df = pd.DataFrame(trans_prob_mat)
        # plt.figure()
        # ax = sns.heatmap(df, cmap='jet', cbar=False)
        # plt.xticks(alpha=0)
        # plt.tick_params(axis='x', width=0)
        # plt.yticks(alpha=0)
        # plt.tick_params(axis='y', width=0)
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # img_path = '{}_img_fea_avenage.jpg'.format(self.global_num)
        # plt.savefig(os.path.join(_save_path,img_path), transparent=True)   
        # for k in range(len(_fea)):
        #     if k % 40 == 0:
        #         fea_1 = _fea[k,:,:]
        #         _a = np.clip(fea_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        #         trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        #         df = pd.DataFrame(trans_prob_mat)
        #         plt.figure()
        #         ax = sns.heatmap(df, cmap='jet', cbar=False)
        #         plt.xticks(alpha=0)
        #         plt.tick_params(axis='x', width=0)
        #         plt.yticks(alpha=0)
        #         plt.tick_params(axis='y', width=0)
        #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #         plt.margins(0, 0)
        #         img_path = '{}_img_fea_{}.jpg'.format(self.global_num,k)
        #         plt.savefig(os.path.join(_save_path,img_path), transparent=True)   



        x = self.up3(x, sfs[1])# x=[1,256,128,128]
        # # ######################################################
        # _save_path = os.path.join(self.save_path,'up3')
        # if not os.path.exists(_save_path):
        #     os.makedirs(_save_path)
            
        # _fea = x[0,:,:,:].cpu().data.numpy()
        # _fea_mean = np.mean(_fea, axis=0)
        # _a = np.clip(_fea_mean, 0, 1) # 将numpy数组约束在[0, 1]范围内
        # trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        # df = pd.DataFrame(trans_prob_mat)
        # plt.figure()
        # ax = sns.heatmap(df, cmap='jet', cbar=False)
        # plt.xticks(alpha=0)
        # plt.tick_params(axis='x', width=0)
        # plt.yticks(alpha=0)
        # plt.tick_params(axis='y', width=0)
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # img_path = '{}_img_fea_avenage.jpg'.format(self.global_num)
        # plt.savefig(os.path.join(_save_path,img_path), transparent=True)   
        # for k in range(len(_fea)):
        #     if k % 40 == 0:
        #         fea_1 = _fea[k,:,:]
        #         _a = np.clip(fea_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        #         trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        #         df = pd.DataFrame(trans_prob_mat)
        #         plt.figure()
        #         ax = sns.heatmap(df, cmap='jet', cbar=False)
        #         plt.xticks(alpha=0)
        #         plt.tick_params(axis='x', width=0)
        #         plt.yticks(alpha=0)
        #         plt.tick_params(axis='y', width=0)
        #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #         plt.margins(0, 0)
        #         img_path = '{}_img_fea_{}.jpg'.format(self.global_num,k)
        #         plt.savefig(os.path.join(_save_path,img_path), transparent=True)   


        
        
        
        x = self.up4(x, sfs[0])# x=[1,256,256,256]
        # # ######################################################
        # _save_path = os.path.join(self.save_path,'up4')
        # if not os.path.exists(_save_path):
        #     os.makedirs(_save_path)
            
        # _fea = x[0,:,:,:].cpu().data.numpy()
        # _fea_mean = np.mean(_fea, axis=0)
        # _a = np.clip(_fea_mean, 0, 1) # 将numpy数组约束在[0, 1]范围内
        # trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        # df = pd.DataFrame(trans_prob_mat)
        # plt.figure()
        # ax = sns.heatmap(df, cmap='jet', cbar=False)
        # plt.xticks(alpha=0)
        # plt.tick_params(axis='x', width=0)
        # plt.yticks(alpha=0)
        # plt.tick_params(axis='y', width=0)
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # img_path = '{}_img_fea_avenage.jpg'.format(self.global_num)
        # plt.savefig(os.path.join(_save_path,img_path), transparent=True)   
        # for k in range(len(_fea)):
        #     if k % 40 == 0:
        #         fea_1 = _fea[k,:,:]
        #         _a = np.clip(fea_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        #         trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        #         df = pd.DataFrame(trans_prob_mat)
        #         plt.figure()
        #         ax = sns.heatmap(df, cmap='jet', cbar=False)
        #         plt.xticks(alpha=0)
        #         plt.tick_params(axis='x', width=0)
        #         plt.yticks(alpha=0)
        #         plt.tick_params(axis='y', width=0)
        #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #         plt.margins(0, 0)
        #         img_path = '{}_img_fea_{}.jpg'.format(self.global_num,k)
        #         plt.savefig(os.path.join(_save_path,img_path), transparent=True)   






        x = self.up5(x)         # x=[1,32,512,512]
        ###################################################################
        ######################################################
        # _save_path = os.path.join(self.save_path,'up5')
        # # _save_path='/data/chengzhicao/VLM/VPTTA-main/OPTIC/visual_feature/up5'
        # if not os.path.exists(_save_path):
        #     os.makedirs(_save_path)
            
        # _fea = x[0,:,:,:].cpu().data.numpy()
        # _fea_mean = np.mean(_fea, axis=0)
        # _a = np.clip(_fea_mean, 0, 1) # 将numpy数组约束在[0, 1]范围内
        # trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        # df = pd.DataFrame(trans_prob_mat)
        # plt.figure()
        # ax = sns.heatmap(df, cmap='jet', cbar=False)
        # plt.xticks(alpha=0)
        # plt.tick_params(axis='x', width=0)
        # plt.yticks(alpha=0)
        # plt.tick_params(axis='y', width=0)
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # img_path = '{}_img_fea_avenage.jpg'.format(self.global_num)
        # plt.savefig(os.path.join(_save_path,img_path), transparent=True)   
        # for k in range(len(_fea)):
        #     if k % 8 == 0:
        #         fea_1 = _fea[k,:,:]
        #         _a = np.clip(fea_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        #         trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        #         df = pd.DataFrame(trans_prob_mat)
        #         plt.figure()
        #         ax = sns.heatmap(df, cmap='jet', cbar=False)
        #         plt.xticks(alpha=0)
        #         plt.tick_params(axis='x', width=0)
        #         plt.yticks(alpha=0)
        #         plt.tick_params(axis='y', width=0)
        #         plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #         plt.margins(0, 0)
        #         img_path = '{}_img_fea_{}.jpg'.format(self.global_num,k)
        #         plt.savefig(os.path.join(_save_path,img_path), transparent=True)   

            
        
        
        head_input = F.relu(self.bnout(x))# head_input=[1,32,512,512]

        seg_output = self.seg_head(head_input)


        self.global_num = self.global_num + 1
        print('self.global_num=',self.global_num)
        return seg_output, sfs, head_input

    def close(self):
        for sf in self.sfs:
            sf.remove()


# if __name__ == "__main__":
#     model = ResUnet(resnet='resnet34', num_classes=2, pretrained=False)
#     print(model.res)
#     model.cuda().eval()
#     input = torch.rand(2, 3, 512, 512).cuda()
#     seg_output, x_iw_list, iw_loss = model(input)
#     print(seg_output.size())

