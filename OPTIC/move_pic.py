import os
import shutil

# RIM_ONE_r3_data= (99, 256)
# RIM_ONE_r3_data_prompt= (99, 256)
# REFUGE_data= (320, 256)
# ORIGA_data= (500, 256)
# REFUGE_Valid_data= (800, 256)
# Drishti_GS_data= (50, 256)


origin_path = '/data/chengzhicao/VLM/VPTTA-main/visualization_segmentation_results_GNN_prompt_OPTICv1/OPTIC_lr0.001_memory_size40_neighbor16_optimizerAdam_threshold0.4_iters1_inn_num8_Source_Dataset_ORIGA_Target_Dataset_RIM_ONE_r3'
output_path = '/data/chengzhicao/VLM/VPTTA-main/visualization_segmentation_results_GNN_prompt_OPTICv1/Source_Dataset_ORIGA_Target_Dataset_RIM_ONE_r3'
if not os.path.exists(output_path):
    os.makedirs(output_path)

_list = os.listdir(origin_path)

_list.sort()

for i in range(len(_list)):
    img_path = os.path.join(origin_path,_list[i])
    out_img_path = os.path.join(output_path,_list[i])
    if 'src_in_trg' in img_path:
        shutil.copyfile(img_path,out_img_path)
