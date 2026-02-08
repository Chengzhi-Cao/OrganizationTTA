cd /data/chengzhicao/VLM/VPTTA-main/OPTIC
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ batchgenerators medpy

# learning rate
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.1 --memory_size 40 --neighbor 16 --prompt_alpha 1 --warm_n 5 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.05 --memory_size 40 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.01 --memory_size 40 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.005 --memory_size 40 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.0005 --memory_size 40 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.0001 --memory_size 40 --prompt_alpha 1 --inn_num 4


# memory size
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 80 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 70 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 60 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 50 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 30 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 20 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 10 --prompt_alpha 1 --inn_num 4

# neighbor
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --neighbor 24 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --neighbor 20 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --neighbor 12 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --neighbor 8 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --neighbor 4 --prompt_alpha 1 --inn_num 4

# GNN - edge
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --neighbor 24 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --neighbor 20 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --neighbor 12 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --neighbor 8 --prompt_alpha 1 --inn_num 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --neighbor 4 --prompt_alpha 1 --inn_num 4

# --num_ViGBlocks
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 1
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 2
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 3
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 5
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 6
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 7

# num_edge
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 1 --num_edge 9
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 1 --num_edge 8
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 1 --num_edge 7
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 1 --num_edge 6
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 1 --num_edge 5
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 1 --num_edge 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 1 --num_edge 3


python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 2 --num_edge 9
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 2 --num_edge 8
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 2 --num_edge 7
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 2 --num_edge 6
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 2 --num_edge 5
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 2 --num_edge 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 2 --num_edge 3

python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 3 --num_edge 9
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 3 --num_edge 8
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 3 --num_edge 7
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 3 --num_edge 6
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 3 --num_edge 5
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 3 --num_edge 4
python vptta_examv7.py --Source_Dataset Drishti_GS --lr 0.001 --memory_size 40 --prompt_alpha 1 --inn_num 4 --num_ViGBlocks 3 --num_edge 3