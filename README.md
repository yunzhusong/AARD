# Adversary-Aware Rumor Detection (AARD)
![](https://img.shields.io/badge/license-MIT-brightgreen) ![](https://img.shields.io/badge/Python-3.6-blue) ![](https://img.shields.io/badge/Pytorch-1.3.0-orange)

<img src="https://user-images.githubusercontent.com/45812808/118956880-96206000-b992-11eb-8c9a-9d5e865ff554.png" width="800"/>

[[Paper]()][[Dataset](https://drive.google.com/drive/folders/16gTXrGsQx3hcr-_sSdxFnjfdFw0mPCTp?usp=sharing)]

If you use any source code or dataset included in this repo, please cite this paper:
```
@inproceedings{song2021adversary,
  title={Adversary-Aware Rumor Detection},
  author={Song, Yun-Zhu and Chen, Yi-Syuan and Chang, Yi-Ting and Weng, Shao-Yu and Shuai, Hong-Han},
  booktitle={ACL-IJCNLP: Findings},
  year={2021}
}
```

## Introdiction
Many rumor detection models have been proposed to automatically detect the rumors based on the contents and propagation path. However, most previous works are not aware of malicious attacks, e.g., framing. Therefore, we propose a novel rumor detection framework, Adversary-Aware Rumor Detection, to improve the vulnerability of detection models, including Weighted-Edge Transformer-Graph Network and Position-aware Adversarial Response Generator. To the best of our knowledge, this is the first work that can generate the adversarial response with the consideration of the response position. **Even without the adversarial learning process, our detection model (Weighted-Edge Transformer-Graph Network) is also a strong baseline for rumor detection task on Twitter15, Twitter16 and Pheme.**

## Performance
<img src="https://user-images.githubusercontent.com/45812808/118972894-a7259d00-b9a3-11eb-8435-44e177a854e9.png" width="800">

## Getting started

### Requirements
Detailed env is included in ```requirement.txt```

### Dataset and Model Preparation

1. We collect the user comments following Twitter's policy, and the processed dataset is available [here](https://drive.google.com/drive/folders/16gTXrGsQx3hcr-_sSdxFnjfdFw0mPCTp?usp=sharing). The dataset should be placed in ```./dataset/```
2. To train the generator, we need the pretrained model, which can be downloaded [here](https://drive.google.com/file/d/19chy9NMNAIkvWfdoR7-nthg5z8Mu6KKB/view?usp=sharing)). The pretrained generation model should be placed in ```./results/pretrain/```

The data preprocessing is followed [BiGAN](https://github.com/TianBian95/BiGCN). The raw datasets except the comments can be downloaded in [raw_pheme](https://figshare.com/articles/dataset/PHEME_dataset_of_rumours_and_non-rumours/4010619) provided by [Zubiagaet al., 2016](https://arxiv.org/abs/1610.07363) and [raw_twitter15_twitter16](https://www.dropbox.com/s/7ewzdrbelpmrnxu/rumdetect2017.zip?dl=0) provided by [Ma et al., 2017](https://www.aclweb.org/anthology/P17-1066/).

### Code structure
```
|_src\
      |_run.sh  -> script to run the code
      |_main.py
      |_models\
            |_trainer_gen.py    -> warpping different experiments
            |_trainer.py        -> model trainer
            |_model.py          -> main class for AARD model
            |_model_detector.py -> for supporting model.py
            |_model_decoder.py  -> for supporting model.py
            |_predictor.py      -> for decoding form generator
      |_data\   -> for spliting 5-fold and building datagraph
      |_eval\   -> define evaluation metric (Recall, Precision and F-score of each class)
      |_others\ -> define loss, logging info
|_dataset\
      |_Pheme\
      |_Phemetextgraph\     -> can be automatically generated data/getgraph.py
      |_twitter15\
      |_twitter15textgraph\ -> can be automatically generated data/getgraph.py
      |_twitter16\
      |_twitter16textgraph\ -> can be automatically generated data/getgraph.py
|_results\
      |_pretrain\
            |_XSUM_BertExtAbs\
```
### How to run the code

#### Three-stage training
```
python main.py \
  -train_detector \
  -train_adv \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/Pheme' \
  -savepath '../results/Pheme' \
  -batch_size 48 \
  -filter True \
  -log_tensorboard \
  -warmup_steps 100 \
```

#### Only train detector
```
python main.py \
  -train_detector \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/Pheme' \
  -savepath '../results/Pheme' \
  -filter True \
  -batch_size 48 \
  -log_tensorboard \
  -warmup_steps 100 \
```

#### Evaluate detector
```
python main.py \
  -test_detector \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/Pheme' \
  -savepath '../results/Pheme' \
  -filter True \
  -batch_size 48 \
  -log_tensorboard \
  -warmup_steps 100 \
```
#### Evaluate detector under adversarial attack
```
python main.py \
  -test_detector \
  -test_gen \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/Pheme' \
  -savepath '../results/Pheme' \
  -filter True \
  -batch_size 48 \
  -log_tensorboard \
  -warmup_steps 100 \
```

### Other experiments in paper

#### Early rumor detection (only testing)
Run the model testing under different data time stamp.
```
python main.py \
  -early '0,6,12,18,24,30,36,42,48,54,60,120' \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/twitter15' \
  -savepath '../results/twitter15/early_detection' \
  -filter True \
  -batch_size 48 \
```
```
python main.py \
  -early '0,6,12,18,24,30,36,42,48,54,60,120' \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/twitter16' \
  -savepath '../results/twitter16/early_detection' \
  -filter True \
  -batch_size 48 \
```
```
python main.py \
 -early '0,60,120,240,480,720,1440,2880' \
 -fold '0,1,2,3,4' \
 -dataset_dir '../dataset/Pheme' \
 -savepath '../results/Pheme/early_detection' \
 -filter True \
 -batch_size 48 \
 ```
 
#### Data scarcity experiment (need training)
Train the models under different quantities of data, ranging from 5% to100%, and evaluate them on the same testing set.
```
python main.py \
  -train_detector \
  -quat '5,10,25,50,75,100' \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/Pheme' \
  -savepath '../results/pheme/data_scarcity' \
  -filter True \
  -batch_size 48 \
  -log_tensorboard \
```
