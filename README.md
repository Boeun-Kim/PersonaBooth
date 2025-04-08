# PersonaBooth (CVPR 2025)

This is the official implementation of "PersonaBooth: Personalized Text-to-Motion Generation (CVPR 2025)" [[paper]](https://arxiv.org/abs/2503.07390) [[project]](https://boeun-kim.github.io/page-PersonaBooth/)

<br>

![framework](https://github.com/Boeun-Kim/PersonaBooth/blob/main/figures/framework.png)


<br>

## Prepare Codes and Datasets

 ### [ Environments ]

We tested our code on the following environment.

- CUDA 12.1

- python 3.11.0

  

Install python libraries with:

```
pip install -r requirements.txt
```



### [ Download PerMo Dataset ]

PerMO dataset can be downloaded from the [dataset page](https://github.com/AIRC-KETI-VISION/PerMo-dataset/).
1. Download **Mesh (SMPL-H.zip)** and follow the instructions in the "Data preprocess for PersonaBooth" section.
   Save the data in the `data/dataset/smpl` directory.
2. Download **Text Description (PerMo_description.zip)** and unzip in the `data/dataset/description` directory.

   


### [ Prepare Dependencies ]

PersonaBooth is dependent on external modules: [HumanML3D](https://github.com/EricGuo5513/HumanML3D), [MDM](https://github.com/GuyTevet/motion-diffusion-model), and [TMR](https://github.com/Mathux/TMR).

The modified codes are in the `dependency` folder, but some files or pretrained weights need to be downloaded from the original repositories.


1. Download the dependency folder from the following link and unzip.
   (Substitute the dependency folder in this repository.)
   
   🔗 https://drive.google.com/file/d/1UzBbxaADnz8KeZmaDiYNEdZLHZWZSSWF/view?usp=sharing
   


3. Download smpl body model by

   ```
   cd prepare
   sh download_bodymodel.sh
   ```



4. Download the pretrained 50-step MDM weight from the original repository.

   You can download it from the following link:

   🔗 https://drive.google.com/file/d/1cfadR1eZ116TIdXK7qDX1RugAerEiJXr/view

   Save the following files in the `dependency/MDM/pretrained/50step` directory:

   🧾 model000750000.pt

   🧾 args.json

   

5. Download the statistics files of MDM from

   🔗 https://github.com/GuyTevet/motion-diffusion-model/tree/main/dataset

   Save the following files in the `dependency/MDM/dataset` directory:

   🧾 t2m_mean.npy

   🧾 t2m_std.npy
   
   🧾 humanml_opt.txt
   



### [ Prepare Evaluation Datasets and Codes ]

1. Download HumanML3D dataset for evaluation.

   Follow guidelines in the original repository 🔗 [HumanML3D](https://github.com/EricGuo5513/HumanML3D) to obtain preprocessed dataset.

   Save the following file and folders in the `dependency/MDM/HumanML3D` directory:

   🧾 Mean.npy
   
   🧾 Std.npy
   
   🧾 test.txt
   
   📁 new_joint_vecs
   
   📁 texts

   

3. Download PRA classifier for PRA metric by

   ```
   cd prepare
   sh download_PRAeval.sh
   ```

<br>

## Preprocess PerMo Dataset

1. Preprocess the smpl data to obtain guo features, which are the input motion format for our model.

   smpl ➡️ position ➡️ guo feature

   ```
   cd dataset/data
   sh preprocess_PerMo.sh
   ```

<br>

## Download Pretrained Weights

1. **Weights of Motion Clip**

   The Motion CLIP in our model is based on the TMR framework, but the text encoder is replaced with the CLIP text encoder. We provide the pretrained weights for this Motion CLIP.

2. **Weights of PersonaBooth**

   The pretrained weights for PersonaBooth, excluding Motion CLIP and Motion Diffusion (MDM) modules, are provided.

   

   Both checkpoints can be downloaded by

   ```
   cd prepare
   sh download_pretrained.sh
   ```

   
<br>

## Demo - Generate Personalized Motion

```
sh demo.sh
```

1. Place one or more input motions (guo features) in the 'demo/input' directory.
2. Provide the text using the '--text' option. Place 'sks' in the position that modifies the subject. For example: 'sks person walks in a circle.'
3. Users can adjust the hyperparameters in `demo.sh` to find a balance between reflecting the persona and aligning the text.
4. The generated motions can be found in the `demo/result` directory.


<br>

## Train

```
sh train.sh
```

Training arguments can be modified from `arguments_PerMo.py`.


<br>

## Evaluate

```
sh eval.sh
```

The results can be found in the `eval_result` directory.

Random sampling and iteration are performed for the evaluation metrics. Please follow the settings in `eval_humanml3d_metrics.py` and `eval_PRA.py` for the PerMo benchmark.


<br>

## Citation

```
@article{kim2025personabooth,
  title={PersonaBooth: Personalized Text-to-Motion Generation},
  author={Kim, Boeun and Jeong, Hea In and Sung, JungHoon and Cheng, Yihua and Lee, Jeongmin and Chang, Ju Yong and Choi, Sang-Il and Choi, Younggeun and Shin, Saim and Kim, Jungho and Chang, Hyung Jin},
  journal={arXiv preprint arXiv:2503.07390},
  year={2025}
}
```
