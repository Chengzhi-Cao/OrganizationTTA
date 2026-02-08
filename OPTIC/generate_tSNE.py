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
from utils.prompt_myv5_visual import Promptv5 as Prompt

from utils.metrics import calculate_metrics,calculate_metrics_save_segment_output
# from networks.ResUnet_TTA import ResUnet
from networks.ResUnet_TTA_tsne import ResUnet
from torch.utils.data import DataLoader
from dataloaders.OPTIC_dataloader_visual import OPTIC_dataset
from dataloaders.transform import collate_fn_wo_transform
from dataloaders.convert_csv_to_list import convert_labeled_list


torch.set_num_threads(1)


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

        # Optimizer
        self.optim = config.optimizer
        self.lr = config.lr
        self.weight_decay = config.weight_decay
        self.momentum = config.momentum
        self.betas = (config.beta1, config.beta2)
        self.save_results_path = config.save_results_path
        self.threshold_my = config.threshold

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
        self.prompt = Prompt(prompt_alpha=self.prompt_alpha, image_size=self.image_size,save_results_path= self.save_results_path).to(self.device)    # image_size=512
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
                _, low_freq = self.prompt(x)
                init_data, score = self.memory_bank.get_neighbours(keys=low_freq.cpu().detach().numpy(), k=self.neighbor) # k=16
            else:
                init_data = torch.ones((1, 3, self.prompt.prompt_size, self.prompt.prompt_size)).data   #  init_data=[1,3,5,5], self.prompt.prompt_size=5
            self.prompt.update(init_data)


            # Train Prompt for n iters (1 iter in our VPTTA)
            for tr_iter in range(self.iters):   # self.iters=1
                prompt_x, _ = self.prompt(x)    # x=[1,3,512,512], prompt_x=[1,3,512,512]是x的低频区域还原的图片
                self.model(prompt_x)
                times, bn_loss = 0, 0
                for nm, m in self.model.named_modules():
                    if isinstance(m, AdaBN):
                        bn_loss += m.bn_loss
                        times += 1
                loss = bn_loss / times

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.model.change_BN_status(new_sample=False)


            # Inference
            self.model.eval()
            self.prompt.eval()
            with torch.no_grad():
                prompt_x, low_freq = self.prompt(x)                 # x=[1,3,512,512],prompt_x=[1,3,512,512],low_freq=[1,3,5,5]
                pred_logit, fea, head_input = self.model(prompt_x)  # pred_logit=[1,2,512,512],head_input=[1,32,512,512]

            # Update the Memory Bank
            self.memory_bank.push(keys=low_freq.cpu().numpy(), logits=self.prompt.data_prompt.detach().cpu().numpy())# self.prompt.data_prompt=[1,3,5,5]

            # Calculate the metrics
            seg_output = torch.sigmoid(pred_logit)  # seg_output=[1,2,512,512], pred_logit=[1,2,512,512]
            # metrics = calculate_metrics(seg_output.detach().cpu(), y.detach().cpu())
            
            metrics = calculate_metrics_save_segment_output(self.threshold_my,x.detach().cpu(),seg_output.detach().cpu(), y.detach().cpu(),_flag,self.save_results_path)
            
            
            for i in range(len(metrics)):
                assert isinstance(metrics[i], list), "The metrics value is not list type."
                metrics_test[i] += metrics[i]
            
            
            _flag = _flag + 1

        test_metrics_y = np.mean(metrics_test, axis=1)
        print_test_metric_mean = {}
        for i in range(len(test_metrics_y)):
            print_test_metric_mean[metric_dict[i]] = test_metrics_y[i]
        print("Test Metrics Mean: ", print_test_metric_mean)

import cv2 as cv
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    model = ResUnet(resnet='resnet34', num_classes=2, pretrained=False, newBN=AdaBN, warm_n=16).to('cuda:0')
    path1 = '/data/chengzhicao/VLM/TTDG/datasets/Fundus/RIM_ONE_r3/train/image'
    RIM_ONE_r3_data = []
    img_list = os.listdir(path1)
    img_list.sort()
    for i in range(len(img_list)):
        img_path = os.path.join(path1,img_list[i])
        _img = cv.imread(img_path)        
        _img = cv.resize(_img,(512,512))
        _img = torch.from_numpy(_img).to('cuda:0')
        # 简写方式：使用 permute
        images_nhwc = _img.permute(2,0,1)  # 等价于两次 transpose
        # print('images_nhwc=',images_nhwc.shape)
        images_nhwc = images_nhwc.unsqueeze(dim=0)
        images_nhwc = images_nhwc.type(torch.float32)
        _out = model(images_nhwc)
        RIM_ONE_r3_data.append(_out)
    RIM_ONE_r3_data = np.array(RIM_ONE_r3_data)
    print('RIM_ONE_r3_data=',RIM_ONE_r3_data.shape)


    path1 = '/data/chengzhicao/VLM/VPTTA-main/visualization_segmentation_resultsv5_OPTIC_tSNE_99'
    RIM_ONE_r3_data = []
    img_list = os.listdir(path1)
    img_list.sort()
    for i in range(len(img_list)):
        img_path = os.path.join(path1,img_list[i])
        _img = cv.imread(img_path)        
        _img = cv.resize(_img,(512,512))
        _img = torch.from_numpy(_img).to('cuda:0')
        # 简写方式：使用 permute
        images_nhwc = _img.permute(2,0,1)  # 等价于两次 transpose
        # print('images_nhwc=',images_nhwc.shape)
        images_nhwc = images_nhwc.unsqueeze(dim=0)
        images_nhwc = images_nhwc.type(torch.float32)
        _out = model(images_nhwc)
        RIM_ONE_r3_data.append(_out)
    RIM_ONE_r3_data = np.array(RIM_ONE_r3_data)
    RIM_ONE_r3_data_prompt = np.array(RIM_ONE_r3_data)
    print('RIM_ONE_r3_data_prompt=',RIM_ONE_r3_data_prompt.shape)



    path1 = '/data/chengzhicao/VLM/TTDG/datasets/Fundus/REFUGE/train/image'
    REFUGE_data = []
    img_list = os.listdir(path1)
    img_list.sort()
    for i in range(len(img_list)):
        img_path = os.path.join(path1,img_list[i])
        _img = cv.imread(img_path)        
        _img = cv.resize(_img,(512,512))
        _img = torch.from_numpy(_img).to('cuda:0')
        images_nhwc = _img.permute(2,0,1)  # 等价于两次 transpose
        images_nhwc = images_nhwc.unsqueeze(dim=0)
        images_nhwc = images_nhwc.type(torch.float32)
        _out = model(images_nhwc)
        REFUGE_data.append(_out)
    REFUGE_data = np.array(REFUGE_data)
    print('REFUGE_data=',REFUGE_data.shape)







    data1 = RIM_ONE_r3_data # data1=[99,512,256]
    data2 = REFUGE_data     # data2=[320,512,256]
    data3 = RIM_ONE_r3_data_prompt
    
    # 生成三个类别的样本数据的标签
    label1 = np.zeros(data1.shape[0]) + 0
    label2 = np.zeros(data2.shape[0]) + 1
    label3 = np.zeros(data2.shape[0]) + 2
    
    
    data = np.concatenate((data1, data2))
    labels = np.concatenate((label1, label2))
    print(data.shape, labels.shape)

    RANDOM_STATE = 1
    # 使用t-SNE进行降维
    # tsne = TSNE(n_components=2, random_state=42)
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE)
    # tsne = TSNE(n_components=3, random_state=42)
    X_tsne = tsne.fit_transform(data)


    # 归一化
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)

    # 绘制t-SNE可视化图
    plt.figure(figsize=(10, 8))
    plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 图中文字体设置为Times New Roman

    # shape_list = ['o', 'D', '^', 'P']  # 设置不同类别的形状
    shape_list = ['o', 'o', '^', 'P']  # 设置不同类别的形状
    color_list = ['r', 'g', 'b', 'm']  # 设置不同类别的颜色
    label_list = ['Target', 'Source', 'Class 3', 'Class 4']
    # 遍历所有标签种类
    for i in range(len(np.unique(labels))):
        # plt.scatter(X_norm[labels == i, 0], X_norm[labels == i, 1], color=color_list[i],
        #             marker=shape_list[i], s=150, label=label_list[i], alpha=0.5)   # 绘制散点图,s=150表示形状大小, alpha=0.5表示图形透明度(越小越透明)

        plt.scatter(X_norm[labels == i, 0], X_norm[labels == i, 1], color=color_list[i],
                    marker=shape_list[i], s=30, label=label_list[i], alpha=0.5)   # 绘制散点图,s=150表示形状大小, alpha=0.5表示图形透明度(越小越透明)

    # # 遍历所有样本
    color_map = {0:'r', 1:'g', 2:'b', 3:'m'}   # 定义类别颜色映射关系
    # shape_map = {0:'o', 1:'D', 2:'^', 3:'P'}
    shape_map = {0:'o',1:'o' }
    # label_map = {0:'Class 1', 1:'Class 2', 2:'Class 3', 3:'Class 4'}
    label_map = {0:'Source', 1:'Target'}
    for data, label in zip(X_norm, labels):  # 遍历样本数据和标签，根据类别选择颜色并绘制散点图
        plt.scatter(data[0], data[1], color=color_map[label], marker=shape_map[label])
        # plt.text(data[0], data[1], label, ha='center', va='bottom')  # 所有样本都对应写上标签

    # 添加图例，并设置字体大小
    plt.legend(fontsize=10)
    ax = plt.gca()  # gca:get current axis得到当前轴
    # plt.xlabel('t-SNE Dimension 1', fontsize=20)  # 定义坐标轴标题
    # plt.ylabel('t-SNE Dimension 2', fontsize=20)
    # plt.title('t-SNE Visualization', fontsize=24)  # 定义图题
    plt.savefig('/data/chengzhicao/VLM/VPTTA-main/visualization_tSNE_{}.png'.format(RANDOM_STATE))  # 保存图为png格式
    plt.show()










    # input0 = torch.ones((1,3,512,512)).cuda()
    # output0 = model(input0)
    # print('output0=',output0.shape)


    # parser = argparse.ArgumentParser()
    # # Dataset
    # parser.add_argument('--Source_Dataset', type=str, default='REFUGE',
    #                     help='RIM_ONE_r3/REFUGE/ORIGA/REFUGE_Valid/Drishti_GS')
    # parser.add_argument('--Target_Dataset', type=list)
    # parser.add_argument('--threshold', type=float, default=0.4) 

    # parser.add_argument('--num_workers', type=int, default=8)
    # parser.add_argument('--image_size', type=int, default=512)

    # # Model
    # parser.add_argument('--save_results_path', type=str, default='/data/chengzhicao/VLM/VPTTA-main/visualization_segmentation_resultsv2/OPTICv4')
    # parser.add_argument('--backbone', type=str, default='resnet34', help='resnet34/resnet50')
    # parser.add_argument('--in_ch', type=int, default=3)
    # parser.add_argument('--out_ch', type=int, default=2)

    # # Optimizer
    # parser.add_argument('--optimizer', type=str, default='SGD', help='SGD/Adam')
    # parser.add_argument('--lr', type=float, default=0.01,help='0.01')
    # parser.add_argument('--momentum', type=float, default=0.99)  # momentum in SGD
    # parser.add_argument('--beta1', type=float, default=0.9)      # beta1 in Adam
    # parser.add_argument('--beta2', type=float, default=0.99)     # beta2 in Adam.
    # parser.add_argument('--weight_decay', type=float, default=0.00)

    # # Training
    # parser.add_argument('--batch_size', type=int, default=1)
    # parser.add_argument('--iters', type=int, default=4)

    # # Hyperparameters in memory bank, prompt, and warm-up statistics
    # parser.add_argument('--memory_size', type=int, default=40)
    # parser.add_argument('--neighbor', type=int, default=16, help='16')
    # # parser.add_argument('--prompt_alpha', type=float, default=0.01)
    # # parser.add_argument('--prompt_alpha', type=float, default=1/32)
    # parser.add_argument('--prompt_alpha', type=float, default=1)
    # parser.add_argument('--warm_n', type=int, default=16)

    # # Path
    # parser.add_argument('--path_save_log', type=str, default='/data/chengzhicao/VLM/VPTTA-main/OPTIC/logs')
    # parser.add_argument('--model_root', type=str, default='/data/chengzhicao/VLM/VPTTA-main/pretrained_model/OPTIC')
    # parser.add_argument('--dataset_root', type=str, default='/data/chengzhicao/VLM/TTDG/datasets/Fundus')

    # # Cuda (default: the first available device)
    # parser.add_argument('--device', type=str, default='cuda:0')

    # config = parser.parse_args()

    # # config.Target_Dataset = ['RIM_ONE_r3', 'REFUGE', 'ORIGA', 'REFUGE_Valid', 'Drishti_GS']
    # config.Target_Dataset = ['RIM_ONE_r3', 'REFUGE']
    # config.Target_Dataset.remove(config.Source_Dataset)
    # config.save_results_path = '/data/chengzhicao/VLM/VPTTA-main/visualization_segmentation_resultsv4_OPTIC_tSNE/OPTICv4_' + 'lr{}'.format(config.lr) + '_memory_size{}'.format(config.memory_size) +"_neighbor{}".format(config.neighbor) + '_optimizer{}'.format(config.optimizer) + '_threshold{}'.format(config.threshold) + '_iters{}'.format(config.iters) + 'Source_Dataset_{}'.format(config.Source_Dataset)
    # print("--------------------------------------------------------------")
    # print('source_data={},lr={}, memory_size={}, neighbor={}, prompt_alpha={} '.format(config.Source_Dataset,config.lr,config.memory_size,config.neighbor,config.prompt_alpha))
    # TTA = VPTTA(config)
    # TTA.run()




