import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.models.GPPNN import FeatureExtract
# from OPTIC.utils.models.GPPNN import FeatureExtract
from utils.models.model_GNN import VGNN


class Prompt(nn.Module):
    def __init__(self, prompt_alpha=1, image_size=352,inn_num=2,num_ViGBlocks=1):
        super().__init__()
        self.prompt_size = int(image_size * prompt_alpha) if int(image_size * prompt_alpha) > 1 else 1
        self.padding_size = (image_size - self.prompt_size)//2
        self.init_para = torch.ones((1, 3, self.prompt_size, self.prompt_size))
        self.data_prompt = nn.Parameter(self.init_para, requires_grad=True)
        self.pre_prompt = self.data_prompt.detach().cpu().data
        self.inn_num = inn_num
        self.num_ViGBlocks = num_ViGBlocks
        self.model2 = FeatureExtract(block_num=self.inn_num)
        self.VGNN = VGNN(num_patches=484,out_feature=256*3,num_ViGBlocks=self.num_ViGBlocks)
        

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
    #     _, _, imgH, imgW = x.size()

    #     fft = torch.fft.fft2(x.clone(), dim=(-2, -1))

    #     # extract amplitude and phase of both ffts
    #     amp_src, pha_src = torch.abs(fft), torch.angle(fft)
    #     amp_src = torch.fft.fftshift(amp_src)

    #     # obtain the low frequency amplitude part
    #     prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
    #                                       self.padding_size, imgW - self.padding_size - self.prompt_size],
    #                    mode='constant', value=1.0).contiguous()

    #     amp_src_ = amp_src * prompt
    #     amp_src_ = torch.fft.ifftshift(amp_src_)

    #     amp_low_ = amp_src[:, :, self.padding_size:self.padding_size+self.prompt_size, self.padding_size:self.padding_size+self.prompt_size]

    #     src_in_trg = self.iFFT(amp_src_, pha_src, imgH, imgW)
    #     return src_in_trg, amp_low_

    def forward(self, x):   # x=[1,3,512,512]
        _, _, imgH, imgW = x.size()
        # # obtain the low frequency amplitude part
        
        prompt1 = self.model2(x)
        # prompt = F.pad(self.data_prompt, [self.padding_size, imgH - self.padding_size - self.prompt_size,
        #                                   self.padding_size, imgW - self.padding_size - self.prompt_size],
        #                mode='constant', value=1.0).contiguous() # self.data_prompt=[1,3,5,5], prompt=[1,3,512,512]
        
        amp_src = x
        # # prompt = [1,3,512,512]
        # amp_src_ = amp_src * prompt
        amp_src_ = amp_src * prompt1

        with torch.no_grad():
            # amp_low_ = self.conv1(x)    
            amp_low_ = self.VGNN(prompt1)
        
        
        return amp_src_, amp_low_ 
        # src_in_trg=[1,3,352,352]根据低频区域还原的图片, amp_low_=[1,484,768]是原图片的低频区域