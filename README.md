# Adversary-Aware Rumor Detection (AARD)
![](https://img.shields.io/badge/license-MIT-brightgreen) ![](https://img.shields.io/badge/Python-3.6-blue) ![](https://img.shields.io/badge/Pytorch-1.3.0-orange)

<img src="https://user-images.githubusercontent.com/45812808/118956880-96206000-b992-11eb-8c9a-9d5e865ff554.png" width="800"/>

Codes for the paper: Adversary-Aware Rumor Detection \[[paper]()\]

## Introdiction
Many rumor detection models have been proposed to automatically detect the rumors based on the contents and propagation path. However, most previous works are not aware of malicious attacks, e.g., framing. Therefore, we propose a novel rumor detection framework, Adversary-Aware Rumor Detection, to improve the vulnerability of detection models, including Weighted-Edge Transformer-Graph Network and Position-aware Adversarial Response Generator. To the best of our knowledge, this is the first work that can generate the adversarial response with the consideration of the response position. **Even without the adversarial learning process, our detection model (Weighted-Edge Transformer-Graph Network) is also a strong baseline for rumor detection task on Twitter15, Twitter16 and Pheme.**

## Performance
<img src="https://user-images.githubusercontent.com/45812808/118972894-a7259d00-b9a3-11eb-8435-44e177a854e9.png" width="800">

## Getting started

### Requirements

### Code structure
```
|_src\
      |_run.sh  -> script to run the code
      |_main.py -> 
      |_models\
            |_trainer_gen.py -> warpping different experiments
            |_trainer.py -> model trainer
            |_model.py -> main class for AARD model
            |_model_detector.py -> for supporting model.py
            |_model_decoder.py -> for supporting model.py
            |_predictor.py -> for decoding form generator
      |_data\ -> for spliting 5-fold and building datagraph
      |_eval\ -> define evaluation metric (Recall, Precision and F-score of each class)
      |_others\ -> define loss, logging info

|_dataset\
      |_Pheme\
      |_twitter15\
      |_twitter16\
```
### How to run the code

#### Three-stage training
The testing results will come out while trianing the models.
```
python main.py \
  -train_detector \
  -train_adv \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/Pheme' \
  -savepath '../results/Pheme' \
  -batch_size 48 \
  -filter True \
  -train_epoch 40 \
  -log_tensorboard \
  -warmup_steps 100 \
```

#### Only train detector
The testing results will come out while trianing the models.
```
python main.py \
  -train_detector \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/Pheme' \
  -savepath '../results/Pheme' \
  -filter True \
  -batch_size 48 \
  -train_epoch 40 \
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
  -train_epoch 40 \
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
  -train_epoch 40 \
  -log_tensorboard \
  -warmup_steps 100 \
```


### Other experiments in paper

#### early rumor detection
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
#### data scarcity test
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
