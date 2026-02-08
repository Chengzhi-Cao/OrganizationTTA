import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import scipy.fftpack as fp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import random
import os
import numpy as np
from utils.models.GPPNN import FeatureExtract
# from OPTIC.utils.models.GPPNN import FeatureExtract
from utils.models.model_GNN import VGNN
from utils.models.GPPNN import *
class Prompt(nn.Module):
    def __init__(self, prompt_alpha=1/32, image_size=512,inn_num=8,save_results_path=None):
        super().__init__()
        self.save_path1 = save_results_path
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        self.padding_size = (image_size - self.prompt_size)//2
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data


        self.inn_num = inn_num
        self.save_path1 = save_results_path
        
        v = torch.ones((352,352))
        self.v = nn.Parameter(v)
        self.global_num = 0


        self.model2 = FeatureExtract(channel_in=3, channel_split_num=2, subnet_constructor=subnet('DBNet'), block_num=self.inn_num)
        self.VGNN = VGNN(num_patches=484,out_feature=256*3)
        
 
    def update(self, init_data):
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)

        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

    # def forward(self, x):
    #     if not os.path.exists(self.save_path1):
    #         os.makedirs(self.save_path1)
            
            
    #     _x = x[0,:,:,:].cpu().data.numpy() 
    #     _x = 255*(_x - np.min(_x)) /(np.max(_x)-np.min(_x))
    #     _x =  np.uint8(_x)
    #     _x = _x.transpose(1,2,0)
    #     _x = Image.fromarray(_x)
    #     img_x_path = '{}_x.jpg'.format(self.global_num)
    #     _x.save(os.path.join(self.save_path1,img_x_path))
        
    #     # _x = x[0,:,:,:].cpu().data.numpy() 
    #     # _x = np.mean(_x, axis=0)
    #     # freq = fp.fft2(_x)
    #     # freq2 = fp.fftshift(freq)  # 移位变换系数，使得直流分量在中间
    #     # img_mag = 20 * np.log10(0.1 + np.abs(freq2))
    #     # img_phase = np.angle(freq2)
    #     # img_phase = 255*(img_phase- np.min(img_phase))/(np.max(img_phase)-np.min(img_phase))

    #     # im2 = Image.fromarray(img_mag)



    #     _, _, imgH, imgW = x.size()

    #     fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

    #     # extract amplitude and phase of both ffts
    #     amp_src, pha_src = torch.abs(fft), torch.angle(fft)
        
    #     amp_src = amp_src * self.v
        
    #     amp_src = torch.fft.fftshift(amp_src)

    #     # obtain the low frequency amplitude part
    #     prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
    #                                       self.padding_size, imgW - self.padding_size - self.prompt_size],
    #                    mode='constant', value=1.0).contiguous()



    #     # _fea = prompt[0,:,:,:].cpu().data.numpy() 
    #     # # for k in range(len(_fea)):
    #     # fea_1 = _fea[0,:,:]
    #     # _a = np.clip(fea_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
    #     # trans_prob_mat = (_a.T/np.sum(_a, 1)).T
    #     # df = pd.DataFrame(trans_prob_mat)
    #     # plt.figure()
    #     # # ax = sns.heatmap(df, cmap='jet', cbar=False)
    #     # ax = sns.heatmap(df, cmap='viridis', cbar=False)
    #     # plt.xticks(alpha=0)
    #     # plt.tick_params(axis='x', width=0)
    #     # plt.yticks(alpha=0)
    #     # plt.tick_params(axis='y', width=0)
    #     # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    #     # plt.margins(0, 0)
    #     # img_path = '{}_prompt.jpg'.format(self.global_num)
    #     # plt.savefig(os.path.join(self.save_path1,img_path), transparent=True)   
    #     # plt.close()


    #     amp_src_ = amp_src * prompt
    #     amp_src_ = torch.fft.ifftshift(amp_src_)

    #     amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]

    #     src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)


    #     # _src_in_trg = src_in_trg[0,:,:,:].cpu().data.numpy() 
    #     # _src_in_trg = 255*(_src_in_trg - np.min(_src_in_trg)) /(np.max(_src_in_trg)-np.min(_src_in_trg))
    #     # _src_in_trg =  np.uint8(_src_in_trg)
    #     # _src_in_trg = _src_in_trg.transpose(1,2,0)
    #     # _src_in_trg = Image.fromarray(_src_in_trg)
    #     # img_src_in_trg_path = '{}_src_in_trg.jpg'.format(self.global_num)
    #     # _src_in_trg.save(os.path.join(self.save_path1,img_src_in_trg_path))


    #     self.global_num = self.global_num + 1

    #     return src_in_trg, amp_low_

    def forward(self, x):   # x=[1,3,512,512]
        if not os.path.exists(self.save_path1):
            os.makedirs(self.save_path1)


        _x = x[0,:,:,:].cpu().data.numpy() 
        _x = 255*(_x - np.min(_x)) /(np.max(_x)-np.min(_x))
        _x =  np.uint8(_x)
        _x = _x.transpose(1,2,0)
        _x = Image.fromarray(_x)
        img_x_path = '{}_x.jpg'.format(self.global_num)
        _x.save(os.path.join(self.save_path1,img_x_path))
        

        _, _, imgH, imgW = x.size()
        # # obtain the low frequency amplitude part
        
        prompt1 = self.model2(x)
        # prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
        #                                   self.padding_size, imgW - self.padding_size - self.prompt_size],
        #                mode='constant', value=1.0).contiguous() # self.data_prompt=[1,3,5,5], prompt=[1,3,512,512]
        
        _x = prompt1[0,:,:,:].cpu().data.numpy() 
        _x = 255*(_x - np.min(_x)) /(np.max(_x)-np.min(_x))
        _x =  np.uint8(_x)
        _x = _x.transpose(1,2,0)
        _x = Image.fromarray(_x)
        img_x_path = '{}_inn_output.jpg'.format(self.global_num)
        _x.save(os.path.join(self.save_path1,img_x_path))
        
        
        
        amp_src = x
        # # prompt = [1,3,512,512]
        # amp_src_ = amp_src * prompt
        amp_src_ = amp_src * prompt1

        with torch.no_grad():
            # amp_low_ = self.conv1(x)    
            amp_low_ = self.VGNN(prompt1)
        

        _src_in_trg = amp_src_[0,:,:,:].cpu().data.numpy() 
        _src_in_trg = 255*(_src_in_trg - np.min(_src_in_trg)) /(np.max(_src_in_trg)-np.min(_src_in_trg))
        _src_in_trg =  np.uint8(_src_in_trg)
        _src_in_trg = _src_in_trg.transpose(1,2,0)
        _src_in_trg = Image.fromarray(_src_in_trg)
        img_src_in_trg_path = '{}_src_in_trg.jpg'.format(self.global_num)
        _src_in_trg.save(os.path.join(self.save_path1,img_src_in_trg_path))

        self.global_num = self.global_num + 1

        return amp_src_, amp_low_ # src_in_trg=[1,3,512,512]根据低频区域还原的图片, amp_low_=[1,3,5,5]是原图片的低频区域
