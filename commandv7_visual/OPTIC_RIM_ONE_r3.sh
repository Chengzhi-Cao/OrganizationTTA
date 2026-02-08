cd /data/chengzhicao/VLM/VPTTA-main/OPTIC
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ batchgenerators medpy
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.1 --memory_size 40 --neighbor 16 --prompt_alpha 1 --warm_n 5 --inn_num 8
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.05 --memory_size 40 --prompt_alpha 1 --inn_num 8
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.01 --memory_size 40 --prompt_alpha 1 --inn_num 8
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.005 --memory_size 40 --prompt_alpha 1 --inn_num 8
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 8
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.0005 --memory_size 40 --prompt_alpha 1 --inn_num 8
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.0001 --memory_size 40 --prompt_alpha 1 --inn_num 8


python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.05 --memory_size 30 --prompt_alpha 1 --inn_num 8
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.05 --memory_size 20 --prompt_alpha 1 --inn_num 8
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.05 --memory_size 10 --prompt_alpha 1 --inn_num 8


python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 8 --num_ViGBlocks 2
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 8 --num_ViGBlocks 4
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 8 --num_ViGBlocks 6
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 6 --num_ViGBlocks 2
python vptta_examv6_visual.py --Source_Dataset RIM_ONE_r3 --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 2