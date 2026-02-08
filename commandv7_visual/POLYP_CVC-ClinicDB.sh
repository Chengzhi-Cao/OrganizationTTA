cd /data/chengzhicao/VLM/VPTTA-main/POLYP
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ batchgenerators medpy
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --neighbor 16 --prompt_alpha 0.01 --warm_n 5
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.05 --memory_size 40
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.01 --memory_size 40
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.005 --memory_size 40
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.001 --memory_size 40
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.0005 --memory_size 40
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40


python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.05 --memory_size 30
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.05 --memory_size 20
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.05 --memory_size 10

python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --neighbor 24
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --neighbor 20
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --neighbor 12
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --neighbor 8
python vptta_examv6_visual.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --neighbor 4

