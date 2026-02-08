cd /data/chengzhicao/VLM/VPTTA-main/POLYP
pip install -i https://pypi.mirrors.ustc.edu.cn/simple/ batchgenerators medpy


python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 1 --optimizer Adam 
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 1 --optimizer SGD
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 1 --optimizer AdamW


python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 1 --optimizer Adam 
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.9 --optimizer Adam 
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.8 --optimizer Adam 
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.7 --optimizer Adam 
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.6 --optimizer Adam
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.5 --optimizer Adam 
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.4 --optimizer Adam
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.3 --optimizer Adam 
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.2 --optimizer Adam
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.1 --optimizer Adam
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.05 --optimizer Adam
python vptta_exam_freqv1.py --Source_Dataset CVC-ClinicDB --lr 0.0001 --memory_size 40 --prompt_alpha 1 --neighbor 16 --prompt_alpha 0.01 --optimizer Adam
