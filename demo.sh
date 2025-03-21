python3 demo.py --checkpoint_path 'pretrained/PerMo_checkpoint.pt' \
--w_txt 0.5 --guide_txt 10 --token_renorm 0.3 --guide_motion 15 --motion_renorm 0.3 \
--text 'sks person walks in a circle.'

# In paper, we set w_txt=0.7 for a single input and w_txt=0.5 for multiple inputs. 
# User can find balance between presona reflection and text alignment by adjusting hyperparameters.