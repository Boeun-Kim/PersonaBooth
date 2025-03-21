
# In paper, we set w_txt=0.7 for a single input and w_txt=0.5 for multiple inputs. 

mkdir -p eval_result

# Multiple Input (MI) setting

############################################

python3 eval_humanml3d_metrics.py --checkpoint_path 'pretrained/PerMo_checkpoint.pt' \
--MI_setting True \
--w_txt 0.5 --guide_txt 10 --token_renorm 0.3 --guide_motion 15 --motion_renorm 0.3

python3 eval_PRA.py --checkpoint_path 'pretrained/PerMo_checkpoint.pt' \
--MI_setting True \
--w_txt 0.5 --guide_txt 10 --token_renorm 0.3 --guide_motion 15 --motion_renorm 0.3

############################################



# Single Input (SI) setting

############################################

#python3 eval_humanml3d_metrics.py  --checkpoint_path 'pretrained/PerMo_checkpoint.pt' \
#--w_txt 0.7 --guide_txt 10 --token_renorm 0.3 --guide_motion 15 --motion_renorm 0.3

#python3 eval_PRA.py --checkpoint_path 'pretrained/PerMo_checkpoint.pt' \
#--w_txt 0.7 --guide_txt 10 --token_renorm 0.3 --guide_motion 15 --motion_renorm 0.3

############################################

