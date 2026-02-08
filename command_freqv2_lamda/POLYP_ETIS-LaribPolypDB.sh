cd /data/chengzhicao/VLM/VPTTA-main/POLYP
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ batchgenerators medpy



python vptta_exam_freqv1.py --Source_Dataset ETIS-LaribPolypDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 1 --optimizer Adam --lamda 1
python vptta_exam_freqv1.py --Source_Dataset ETIS-LaribPolypDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 1 --optimizer Adam --lamda 0.75
python vptta_exam_freqv1.py --Source_Dataset ETIS-LaribPolypDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 1 --optimizer Adam --lamda 0.5
python vptta_exam_freqv1.py --Source_Dataset ETIS-LaribPolypDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 1 --optimizer Adam --lamda 0.25
python vptta_exam_freqv1.py --Source_Dataset ETIS-LaribPolypDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 1 --optimizer Adam --lamda 0.1
python vptta_exam_freqv1.py --Source_Dataset ETIS-LaribPolypDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 1 --optimizer Adam --lamda 0.05
