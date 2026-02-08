# v5 512x512 prompt再加上一个滤波1x1

import os
import torch
import numpy as np
import argparse, sys, datetime
from config import Logger
from torch.autograd import Variable
from utils.convert import AdaBN
from utils.memory import Memory
# from utils.prompt_my import Prompt
# from utils.prompt_myv5 import Promptv5 as Prompt
from utils.prompt_my_freqv1 import Promptv5 as Prompt
from utils.metrics import calculate_metrics
from networks.ResUnet_TTA import ResUnet
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader import OPTIC_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list
import torch.nn.functional as F
import torch.nn as nn

torch.set_num_threads(1)



class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1, contrast_mode='all'):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        
    def forward(self, feature1, feature2):
        """
        Args:
            feature1: [1, 3, 512, 512]
            feature2: [1, 3, 512, 512]
        Returns:
            contrastive_loss: scalar
        """
        # 将特征图展平为特征向量
        batch_size, channels, height, width = feature1.shape
        
        # 方法1: 全局平均池化得到全局特征
        feat1_global = F.adaptive_avg_pool2d(feature1, (1, 1)).view(batch_size, -1)  # [1, 3]
        feat2_global = F.adaptive_avg_pool2d(feature2, (1, 1)).view(batch_size, -1)  # [1, 3]
        
        # 归一化特征
        feat1_global = F.normalize(feat1_global, p=2, dim=1)
        feat2_global = F.normalize(feat2_global, p=2, dim=1)
        
        # 计算相似度矩阵
        similarity_matrix = torch.matmul(feat1_global, feat2_global.T) / self.temperature  # [1, 1]
        
        # 对于batch_size=1的情况，直接最大化相似度
        loss = -similarity_matrix.mean()
        
        return loss
    
    

class VPTTA:
    def __init__(self, config):
        # Save Log
        time_now = datetime.datetime.now().__format__("%Y%m%d_%H%M%S_%f")
        log_root = os.path.join(config.path_save_log, 'VPTTA')
        if not os.path.exists(log_root):
            os.makedirs(log_root)
        log_path = os.path.join(log_root, time_now + '.log')
        sys.stdout = Logger(log_path, sys.stdout)

        # Data Loading
        target_test_csv = []
        for target in config.Target_Dataset:
            if target != 'REFUGE_Valid':
                target_test_csv.append(target + '_train.csv')
                target_test_csv.append(target + '_test.csv')
            else:
                target_test_csv.append(target + '.csv')
        ts_img_list, ts_label_list = convert_labeled_list(config.dataset_root, target_test_csv)# len(ts_img_list)=1951,len(ts_label_list)=1951
        target_test_dataset = OPTIC_dataset(config.dataset_root, ts_img_list, ts_label_list,
                                            config.image_size, img_normalize=True)
        self.target_test_loader = DataLoader(dataset=target_test_dataset,
                                             batch_size=config.batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             drop_last=False,
                                             collate_fn=collate_fn_wo_transform,
                                             num_workers=config.num_workers)
        self.image_size = config.image_size
        self.source_dataset = config.Source_Dataset
        # Model
        self.load_model = os.path.join(config.model_root, str(config.Source_Dataset))# config.Source_Dataset='RIM_ONE_r3'  # Pre-trained Source Model
        self.backbone = config.backbone
        self.in_ch = config.in_ch
        self.out_ch = config.out_ch
        self.lamda = config.lamda

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)
        
        self.ContrastiveLoss = ContrastiveLoss()

        # GPU
        self.device = config.device

        # Warm-up
        self.warm_n = config.warm_n

        # Prompt
        self.prompt_alpha = config.prompt_alpha
        self.iters = config.iters

        # Initialize the pre-trained model and optimizer
        self.build_model()

        # Memory Bank
        self.neighbor = config.neighbor
        self.memory_bank = Memory(size=config.memory_size, dimension=self.prompt.data_prompt.numel())

        # Print Information
        # for arg, value in vars(config).items():
        #     print(f"{arg}: {value}")
        self.print_prompt()
        # print('***' * 20)

    def build_model(self):
        self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size).to(self.device)    # image_size=512
        self.model = ResUnet(resnet=self.backbone, num_classes=self.out_ch, pretrained=False, newBN=AdaBN, warm_n=self.warm_n).to(self.device)
        checkpoint = torch.load(os.path.join(self.load_model, 'last-Res_Unet.pth'))
        self.model.load_state_dict(checkpoint, strict=True)

        if self.optim == 'SGD':
            self.optimizer = torch.optim.SGD(
                self.prompt.parameters(),
                lr=self.lr,
                momentum=self.momentum,
                nesterov=True,
                weight_decay=self.weight_decay
            )
        elif self.optim == 'Adam':
            self.optimizer = torch.optim.Adam(
                self.prompt.parameters(),
                lr=self.lr,
                betas=self.betas,
                weight_decay=self.weight_decay
            )

    def print_prompt(self):
        num_params = 0
        for p in self.prompt.parameters():
            num_params += p.numel()
        print("The number of total parameters: {}".format(num_params))

    def run(self):

        metric_dict = ['Dice', 'Enhanced_Align', 'Structure_Measure']
        _flag = 0
        
        
        # Valid on Target
        metrics_test = [[], [], []]
        for batch, data in enumerate(self.target_test_loader):
            x, y = data['data'], data['mask']
            x = torch.from_numpy(x).to(dtype=torch.float32) # x=[1,3,512,512]
            y = torch.from_numpy(y).to(dtype=torch.float32) # y=[1,2,512,512]

            x, y = Variable(x).to(self.device), Variable(y).to(self.device)

            self.model.eval()
            self.prompt.train()
            self.model.change_BN_status(new_sample=True)


            # Initialize Prompt
            if len(self.memory_bank.memory.keys()) >= self.neighbor:    # self.neighbor=16, self.memory_bank.memory.keys()=41
                _, low_freq,_,_ = self.prompt(x)
                init_data, score = self.memory_bank.get_neighbours(keys=low_freq.cpu().detach().numpy(), k=self.neighbor) # k=16
            else:
                init_data = torch.ones((1, 3, self.prompt.prompt_size, self.prompt.prompt_size)).data   #  init_data=[1,3,5,5], self.prompt.prompt_size=5
            self.prompt.update(init_data)


            # Train Prompt for n iters (1 iter in our VPTTA)
            for tr_iter in range(self.iters):   # self.iters=1
                prompt_x, _,amp_src,amp_srcv2 = self.prompt(x)    # x=[1,3,512,512], prompt_x=[1,3,512,512]是x的低频区域还原的图片
                self.model(prompt_x)
                loss1 = self.ContrastiveLoss(amp_src, amp_srcv2)
                times, bn_loss = 0, 0
                for nm, m in self.model.named_modules():
                    if isinstance(m, AdaBN):
                        bn_loss += m.bn_loss
                        times += 1
                loss = bn_loss / times + self.lamda* loss1

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.change_BN_status(new_sample=False)


            # Inference
            self.model.eval()
            self.prompt.eval()
            with torch.no_grad():
                prompt_x, low_freq,_,_ = self.prompt(x)                 # x=[1,3,512,512],prompt_x=[1,3,512,512],low_freq=[1,3,5,5]
                pred_logit, fea, head_input = self.model(prompt_x)  # pred_logit=[1,2,512,512],head_input=[1,32,512,512]

            # Update the Memory Bank
            self.memory_bank.push(keys=low_freq.cpu().numpy(), logits=self.prompt.data_prompt.detach().cpu().numpy())# self.prompt.data_prompt=[1,3,5,5]

            # Calculate the metrics
            seg_output = torch.sigmoid(pred_logit)  # seg_output=[1,2,512,512], pred_logit=[1,2,512,512]
            metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            
            
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics Mean: ", print_test_metric_mean)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Dataset
    parser.add_argument('--Source_Dataset', type=str, default='RIM_ONE_r3',
                        help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    parser.add_argument('--Target_Dataset', type=list)

    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=512)

    # Model
    parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    parser.add_argument('--in_ch', type=int, default=3)
    parser.add_argument('--out_ch', type=int, default=2)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='Adam', help='SGD/Adam')
    parser.add_argument('--lr', type=float, default=0.05)
    parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    parser.add_argument('--weight_decay', type=float, default=0.00)
    parser.add_argument('--lamda', type=float, default=1)

    # Training
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--iters', type=int, default=1)

    # Hyperparameters in memory bank, prompt, and warm-up statistics
    parser.add_argument('--memory_size', type=int, default=40)
    parser.add_argument('--neighbor', type=int, default=16)
    # parser.add_argument('--prompt_alpha', type=float, default=0.01)
    # parser.add_argument('--prompt_alpha', type=float, default=1/32)
    parser.add_argument('--prompt_alpha', type=float, default=1)
    parser.add_argument('--warm_n', type=int, default=16)

    # Path
    parser.add_argument('--path_save_log', type=str, default='/data/chengzhicao/VLM/VPTTA-main/OPTIC/logs')
    parser.add_argument('--model_root', type=str, default='/data/chengzhicao/VLM/VPTTA-main/pretrained_model/OPTIC')
    parser.add_argument('--dataset_root', type=str, default='/data/chengzhicao/VLM/TTDG/datasets/Fundus')

    # Cuda (default: the first available device)
    parser.add_argument('--device', type=str, default='cuda:0')

    config = parser.parse_args()

    config.Target_Dataset = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    config.Target_Dataset.remove(config.Source_Dataset)

    print("--------------------------------------------------------------")
    print('source_data={},lr={}, memory_size={}, neighbor={}, prompt_alpha={} '.format(config.Source_Dataset,config.lr,config.memory_size,config.neighbor,config.prompt_alpha))
    TTA = VPTTA(config)
    TTA.run()
