# For 3-stage training 
python main.py \
  -train_detector \
  -train_adv \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/Pheme' \
  -savepath '../results/Pheme/aard' \
  -batch_size 4 \
  -filter True \
  -visible_gpu 0 \
  -train_epoch 10 \
  -log_tensorboard \
  -warmup_steps 100 \

python main.py \
  -train_detector \
  -train_adv \
  -fold '0' \
  -dataset_dir '../dataset/twitter15' \
  -savepath '../results/twitter15/aard' \
  -batch_size 4 \
  -filter True \
  -visible_gpu 0 \
  -train_epoch 10 \
  -log_tensorboard \
  -warmup_steps 100 \

python main.py \
  -train_detector \
  -train_adv \
  -fold '0' \
  -dataset_dir '../dataset/twitter16' \
  -savepath '../results/twitter16/aard' \
  -batch_size 1 \
  -filter True \
  -visible_gpu 1 \
  -train_epoch 1 \
  -log_tensorboard \
  -warmup_steps 100 \


# For adv generation and testing
python main.py \
  -test_detector \
  -test_gen \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/Pheme' \
  -savepath '../results/Pheme/aard' \
  -batch_size 48 \
  -filter True \
  -visible_gpu 0 \
  -log_tensorboard \

 
python main.py \
  -test_detector \
  -test_gen \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/twitter15' \
  -savepath '../results/twitter15/aard' \
  -batch_size 48 \
  -filter True \
  -visible_gpu 0 \
  -log_tensorboard \
 

python main.py \
  -test_detector \
  -test_gen \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/twitter16' \
  -savepath '../results/twitter16' \
  -batch_size 48 \
  -filter True \
  -visible_gpu 0 \
  -log_tensorboard \
