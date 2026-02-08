import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import random
import os

from PIL import Image
import scipy.fftpack as fp

class Promptv5(nn.Module):
    def __init__(self, prompt_alpha=1/32, image_size=512,save_results_path=None):
        super().__init__()
        
        # self.save_results_path = save_results_path
        self.prompt_alpha = prompt_alpha
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1  # self.prompt_size=5
        self.padding_size = (image_size - self.prompt_size)//2  # self.padding_size = 253
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size)) # self.init_para=[1,3,5,5]
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)# self.data_prompt=[1,3,5,5]
        self.pre_prompt = self.data_prompt.detach().cpu().data  
        
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1,stride=1, bias=False)
        
        self.save_path1 = save_results_path
        self.global_num = 0
        v = torch.ones((512,512))
        self.v = nn.Parameter(v)

    def update(self, init_data):    # init_data=[1,3,5,5]
        with torch.no_grad():
            self.data_prompt.copy_(init_data)

    def iFFT(self, amp_src_, pha_src, imgH, imgW):
        # recompose fft
        real = torch.cos(pha_src) * amp_src_
        imag = torch.sin(pha_src) * amp_src_
        fft_src_ = torch.complex(real=real, imag=imag)
        src_in_trg = torch.fft.ifft2(fft_src_, dim=(-2, -1), s=[imgH, imgW]).real
        return src_in_trg

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
        
        _x = x[0,:,:,:].cpu().data.numpy() 
        _x = np.mean(_x, axis=0)
        freq = fp.fft2(_x)
        freq2 = fp.fftshift(freq)  # 移位变换系数，使得直流分量在中间
        img_mag = 20 * np.log10(0.1 + np.abs(freq2))
        img_phase = np.angle(freq2)
        img_phase = 255*(img_phase- np.min(img_phase))/(np.max(img_phase)-np.min(img_phase))

        # im2 = Image.fromarray(img_mag)
        # if im2.mode == "F":
        #     im2 = im2.convert('L') 
        # freq22 = Image.fromarray(img_phase)
        # if freq22.mode == "F":
        #     freq22 = freq22.convert('L') 
        # img_mag_path = '{}_img_mag.jpg'.format(self.global_num)
        # img_phase_path = '{}_img_phase.jpg'.format(self.global_num)
        # im2.save(os.path.join(self.save_path1,img_mag_path))
        # freq22.save(os.path.join(self.save_path1,img_phase_path))


        _, _, imgH, imgW = x.size()
        fft = torch.fft.fft2(x.clone(), dim=(-2, -1))   # fft=[1,3,512,512]

        # extract amplitude and phase of both ffts
        amp_src, pha_src = torch.abs(fft), torch.angle(fft) # amp_src=[1,3,512,512], pha_src=[1,3,512,512]

        # img_path = '{}_amp_src.jpg'.format(self.global_num)
        # _img = np.uint8(amp_src[0,0].cpu().numpy()*255)
        # Image.fromarray(_img).save(os.path.join(self.save_path1,img_path))
        
        # img_path = '{}_pha_src.jpg'.format(self.global_num)
        # _img = np.uint8(pha_src[0,0].cpu().numpy()*255)
        # Image.fromarray(_img).save(os.path.join(self.save_path1,img_path))
        
        
        
        # # ######################################################
        # _fea = amp_src[0,:,:,:].cpu().data.numpy() 
        # for k in range(len(_fea)):
        #     fea_1 = _fea[k,:,:]
        #     _a = np.clip(fea_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        #     trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        #     df = pd.DataFrame(trans_prob_mat)
        #     plt.figure()
        #     # ax = sns.heatmap(df, cmap='jet', cbar=False)
        #     ax = sns.heatmap(df, cmap='gray', cbar=False)
        #     plt.xticks(alpha=0)
        #     plt.tick_params(axis='x', width=0)
        #     plt.yticks(alpha=0)
        #     plt.tick_params(axis='y', width=0)
        #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #     plt.margins(0, 0)
        #     img_path = '{}_amp_src_{}.jpg'.format(self.global_num,k)
        #     plt.savefig(os.path.join(self.save_path1,img_path), transparent=True)   
        #     plt.close()

        # # ######################################################
        # _fea = pha_src[0,:,:,:].cpu().data.numpy() 
        # for k in range(len(_fea)):
        #     fea_1 = _fea[k,:,:]
        #     _a = np.clip(fea_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        #     trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        #     df = pd.DataFrame(trans_prob_mat)
        #     plt.figure()
        #     ax = sns.heatmap(df, cmap='gray', cbar=False)
        #     plt.xticks(alpha=0)
        #     plt.tick_params(axis='x', width=0)
        #     plt.yticks(alpha=0)
        #     plt.tick_params(axis='y', width=0)
        #     plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        #     plt.margins(0, 0)
        #     img_path = '{}_pha_src_{}.jpg'.format(self.global_num,k)
        #     plt.savefig(os.path.join(self.save_path1,img_path), transparent=True)   
        #     plt.close()

        # amp_src = self.conv1(amp_src)
        amp_src = amp_src * self.v
        amp_src = torch.fft.fftshift(amp_src)

        # obtain the low frequency amplitude part
        prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
                                          self.padding_size, imgW - self.padding_size - self.prompt_size],
                       mode='constant', value=1.0).contiguous() # self.data_prompt=[1,3,5,5]
        # prompt = [1,3,512,512]
        
        
        # _fea = prompt[0,:,:,:].cpu().data.numpy() 
        # fea_1 = _fea[0,:,:]
        # _a = np.clip(fea_1, 0, 1) # 将numpy数组约束在[0, 1]范围内
        # trans_prob_mat = (_a.T/np.sum(_a, 1)).T
        # df = pd.DataFrame(trans_prob_mat)
        # plt.figure()
        # # ax = sns.heatmap(df, cmap='jet', cbar=False)
        # ax = sns.heatmap(df, cmap='viridis', cbar=False)
        # plt.xticks(alpha=0)
        # plt.tick_params(axis='x', width=0)
        # plt.yticks(alpha=0)
        # plt.tick_params(axis='y', width=0)
        # plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        # plt.margins(0, 0)
        # img_path = '{}_prompt.jpg'.format(self.global_num)
        # plt.savefig(os.path.join(self.save_path1,img_path), transparent=True)   
        # plt.close()

        amp_src_ = amp_src * prompt
        amp_src_ = torch.fft.ifftshift(amp_src_)
        # amp_src_ = [1,3,512,512]
        amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]    # self.padding_size = 253, self.prompt_size=5
        # amp_low_ = [1,3,5,5]
        src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
        # return amp_src_, amp_low_ # src_in_trg=[1,3,512,512]根据低频区域还原的图片, amp_low_=[1,3,5,5]是原图片的低频区域

        _src_in_trg = src_in_trg[0,:,:,:].cpu().data.numpy() 
        _src_in_trg = 255*(_src_in_trg - np.min(_src_in_trg)) /(np.max(_src_in_trg)-np.min(_src_in_trg))
        _src_in_trg =  np.uint8(_src_in_trg)
        _src_in_trg = _src_in_trg.transpose(1,2,0)
        _src_in_trg = Image.fromarray(_src_in_trg)
        img_src_in_trg_path = '{}_src_in_trg.jpg'.format(self.global_num)
        _src_in_trg.save(os.path.join(self.save_path1,img_src_in_trg_path))


        
        self.global_num = self.global_num + 1
        if self.global_num > 199:
            print('success')
        
        return src_in_trg, amp_low_ # src_in_trg=[1,3,512,512]根据低频区域还原的图片, amp_low_=[1,3,5,5]是原图片的低频区域
