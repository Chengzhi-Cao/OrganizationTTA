import torch
import torch.nn as nn
import torch.nn.functional as F
from models.GPPNN_runningtime import FeatureExtract
# from OPTIC.utils.models.GPPNN import FeatureExtract
from models.model_GNN import VGNN
from models.GPPNN_runningtime import *
import os
from PIL import Image

import time

class Promptv6(nn.Module):
    def __init__(self, prompt_alpha=1/32, image_size=512,inn_num=2,save_results_path=None):
        super().__init__()


        self.model2 = FeatureExtract(channel_in=3, channel_split_num=2, subnet_constructor=subnet('DBNet'), block_num=2)
        self.VGNN = VGNN(num_patches=64,out_feature=32)
        
    def forward(self, x):   # x=[1,3,512,512]

        _, _, imgH, imgW = x.size()
        # # obtain the low frequency amplitude part
        
        prompt1 = self.model2(x)    # prompt=[2,3,512,512]
        amp_low_ = self.VGNN(prompt1)
        

        return prompt1 # src_in_trg=[1,3,512,512]根据低频区域还原的图片, amp_low_=[1,3,5,5]是原图片的低频区域


from thop import profile
from thop import clever_format

import torchsummary
import torchvision.models as modelss
from torchsummaryX import summary
input1 = torch.ones(1,3,512,512)
# model2 = FeatureExtract()
# output2 = model2(input1)
# print('output2=',output2.shape)

# model_prompt = Promptv6().to('cuda:0')    # image_size=512
# summary(model_prompt, input1)
model_prompt = Promptv6()
torchsummary.summary(model_prompt, (3, 128, 128),device='cpu')




# from torchstat import stat
# model = Promptv6()
# stat(model, (3, 512, 512))  

# macs, params = profile(model, inputs=(input,))
# macs, params = clever_format([macs, params], “%.3f”)
# print(‘flops=’, macs)
# print(‘params=’, params)


x = torch.randn(1,3,128,128)

macs, params = profile(model_prompt, inputs=(x, ))
print('macs:', macs,'params:', params)  # 16865803776.0 3206976.0
print('--------')
macs, params = clever_format([macs, params], "%.3f")
print('macs:', macs,'params:', params)  # 16.866G 3.207M
# print('输出数据的维度是:', y.size())