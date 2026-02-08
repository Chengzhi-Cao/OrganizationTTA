cd /data/chengzhicao/VLM/VPTTA-main/OPTIC
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ batchgenerators medpy
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.1 --memory_size 40 --neighbor 16 --prompt_alpha 1 --warm_n 5
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.05 --memory_size 40 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.01 --memory_size 40 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.005 --memory_size 40 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.001 --memory_size 40 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.0005 --memory_size 40 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.0001 --memory_size 40 --prompt_alpha 1


python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.05 --memory_size 30 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.05 --memory_size 20 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.05 --memory_size 10 --prompt_alpha 1

python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.1 --memory_size 40 --neighbor 24 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.1 --memory_size 40 --neighbor 20 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.1 --memory_size 40 --neighbor 12 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.1 --memory_size 40 --neighbor 8 --prompt_alpha 1
python vptta_examv5.py --Source_Dataset RIM_ONE_r3 --lr 0.1 --memory_size 40 --neighbor 4 --prompt_alpha 1

