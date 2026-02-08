import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.models.GPPNN import FeatureExtract
# from OPTIC.utils.models.GPPNN import FeatureExtract
from utils.models.model_GNN import VGNN
from utils.models.GPPNN import *
import os
from PIL import Image
import time

class Promptv6(nn.Module):
    def __init__(self, prompt_alpha=1/32, image_size=512,inn_num=2,num_ViGBlocks=1,save_results_path=None):
        super().__init__()
        self.prompt_alpha = prompt_alpha
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1  # self.prompt_size=5
        self.padding_size = (image_size - self.prompt_size)//2  # self.padding_size = 253
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size)) # self.init_para=[1,3,5,5]
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)# self.data_prompt=[1,3,5,5]
        self.pre_prompt = self.data_prompt.detach().cpu().data  
        self.inn_num = inn_num
        self.save_path1 = save_results_path
        self.global_num = 0
        self.num_ViGBlocks = num_ViGBlocks
        # self.conv1 = nn.Conv2d(3, 3, kernel_size=1,stride=1, bias=False)
        
        
        # v = torch.ones((512,512))
        # self.v = nn.Parameter(v)
        
        
        self.model2 = FeatureExtract(channel_in=3, channel_split_num=2, subnet_constructor=subnet('DBNet'), block_num=self.inn_num)
        self.VGNN = VGNN(num_patches=1024,out_feature=256*3,num_ViGBlocks=self.num_ViGBlocks)
        

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
        

        _, _, imgH, imgW = x.size()
        # # obtain the low frequency amplitude part
        
        prompt1 = self.model2(x)
        # prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
        #                                   self.padding_size, imgW - self.padding_size - self.prompt_size],
        #                mode='constant', value=1.0).contiguous() # self.data_prompt=[1,3,5,5], prompt=[1,3,512,512]
        
        
        
        
        # _x = prompt1[0,:,:,:].cpu().data.numpy() 
        # _x = 255*(_x - np.min(_x)) /(np.max(_x)-np.min(_x))
        # _x =  np.uint8(_x)
        # _x = _x.transpose(1,2,0)
        # _x = Image.fromarray(_x)
        # img_x_path = '{}_inn_output.jpg'.format(self.global_num)
        # _x.save(os.path.join(self.save_path1,img_x_path))
        
        
        
        amp_src = x
        # # prompt = [1,3,512,512]
        # amp_src_ = amp_src * prompt
        amp_src_ = amp_src * prompt1

        with torch.no_grad():
            # amp_low_ = self.conv1(x)    
            amp_low_ = self.VGNN(prompt1)
        

        # _src_in_trg = amp_src_[0,:,:,:].cpu().data.numpy() 
        # _src_in_trg = 255*(_src_in_trg - np.min(_src_in_trg)) /(np.max(_src_in_trg)-np.min(_src_in_trg))
        # _src_in_trg =  np.uint8(_src_in_trg)
        # _src_in_trg = _src_in_trg.transpose(1,2,0)
        # _src_in_trg = Image.fromarray(_src_in_trg)
        # img_src_in_trg_path = '{}_src_in_trg.jpg'.format(self.global_num)
        # _src_in_trg.save(os.path.join(self.save_path1,img_src_in_trg_path))

        self.global_num = self.global_num + 1

        return amp_src_, amp_low_ # src_in_trg=[1,3,512,512]根据低频区域还原的图片, amp_low_=[1,3,5,5]是原图片的低频区域
