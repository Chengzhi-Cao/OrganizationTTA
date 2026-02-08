cd /data/chengzhicao/VLM/VPTTA-main/POLYP
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ batchgenerators medpy

python vptta_examv6.py --Source_Dataset BKAI --lr 0.001 --memory_size 40 --num_ViGBlocks 1
python vptta_examv6.py --Source_Dataset BKAI --lr 0.001 --memory_size 40 --num_ViGBlocks 2
python vptta_examv6.py --Source_Dataset BKAI --lr 0.001 --memory_size 40 --num_ViGBlocks 3
python vptta_examv6.py --Source_Dataset BKAI --lr 0.001 --memory_size 40 --num_ViGBlocks 4
python vptta_examv6.py --Source_Dataset BKAI --lr 0.001 --memory_size 40 --num_ViGBlocks 5

print("------------------------------------------------------------------------------------")

python vptta_examv6.py --Source_Dataset BKAI --lr 0.005 --memory_size 40 --num_ViGBlocks 1
python vptta_examv6.py --Source_Dataset BKAI --lr 0.005 --memory_size 40 --num_ViGBlocks 2
python vptta_examv6.py --Source_Dataset BKAI --lr 0.005 --memory_size 40 --num_ViGBlocks 3
python vptta_examv6.py --Source_Dataset BKAI --lr 0.005 --memory_size 40 --num_ViGBlocks 4
python vptta_examv6.py --Source_Dataset BKAI --lr 0.005 --memory_size 40 --num_ViGBlocks 5

print("------------------------------------------------------------------------------------")

python vptta_examv6.py --Source_Dataset BKAI --lr 0.0001 --memory_size 40 --num_ViGBlocks 1
python vptta_examv6.py --Source_Dataset BKAI --lr 0.0001 --memory_size 40 --num_ViGBlocks 2
python vptta_examv6.py --Source_Dataset BKAI --lr 0.0001 --memory_size 40 --num_ViGBlocks 3
python vptta_examv6.py --Source_Dataset BKAI --lr 0.0001 --memory_size 40 --num_ViGBlocks 4
python vptta_examv6.py --Source_Dataset BKAI --lr 0.0001 --memory_size 40 --num_ViGBlocks 5

print("------------------------------------------------------------------------------------")
