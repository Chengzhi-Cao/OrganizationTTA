#!/bin/bash

#Please modify the following roots to yours.
dataset_root=/data/chengzhicao/VLM/TTDG/datasets/Fundus
model_root=/data/chengzhicao/VLM/VPTTA-main/pretrained_model/OPTIC
path_save_log=/data/chengzhicao/VLM/VPTTA-main/OPTIC/logs

#Dataset [RIM_ONE_r3, REFUGE, ORIGA, REFUGE_Valid, Drishti_GS]
Source=RIM_ONE_r3

#Optimizer
optimizer=Adam
lr=0.05

#Hyperparameters
memory_size=40
neighbor=16
prompt_alpha=0.01
warm_n=5

#Command
cd OPTIC
CUDA_VISIBLE_DEVICES=0 python vptta.py \
--dataset_root $dataset_root --model_root $model_root --path_save_log $path_save_log \
--Source_Dataset $Source \
--optimizer $optimizer --lr $lr \
--memory_size $memory_size --neighbor $neighbor --prompt_alpha $prompt_alpha --warm_n $warm_n