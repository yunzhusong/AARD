#export CUDA_VISIBLE_DEVICES=1

# For 3-stage training 
python main.py \
  -train_detector \
  -train_adv \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/Pheme' \
  -batch_size 4 \
  -savepath '../results/Pheme/aard' \
  -filter True \
  -visible_gpu 0 \
  -train_epoch 10 \
  -log_tensorboard \
  -warmup_steps 100 \

 
#python main.py \
#  -train_detector \
#  -train_adv \
#  -fold '0,1,2,3,4' \
#  -dataset_dir '../dataset/twitter15' \
#  -batch_size 4 \
#  -savepath '../results/twitter15/aard' \
#  -filter True \
#  -visible_gpu 0 \
#  -train_epoch 10 \
#  -log_tensorboard \
#  -warmup_steps 100 \
# 
#
#python main.py \
#  -train_detector \
#  -train_adv \
#  -fold '0' \
#  -dataset_dir '../dataset/twitter16' \
#  -batch_size 4 \
#  -savepath '../results/twitter16/aard' \
#  -filter True \
#  -visible_gpu 1 \
#  -train_epoch 10 \
#  -log_tensorboard \
#  -warmup_steps 100 \
#

## For adv generation and testing
#python main.py \
#  -test_detector \
#  -test_gen \
#  -fold '0,1,2,3,4' \
#  -dataset_dir '../dataset/Pheme' \
#  -batch_size 48 \
#  -savepath '../results/Pheme' \
#  -filter True \
#  -visible_gpu 0 \
#  -train_epoch 40 \
#  -log_tensorboard \
#  -warmup_steps 100 \
#
# 
#python main.py \
#  -test_detector \
#  -test_gen \
#  -fold '0,1,2,3,4' \
#  -dataset_dir '../dataset/twitter15' \
#  -batch_size 12 \
#  -savepath '../results/twitter15' \
#  -filter True \
#  -visible_gpu 1 \
#  -train_epoch 40 \
#  -log_tensorboard \
#  -warmup_steps 100 \
 

#python main.py \
#  -test_detector \
#  -test_gen \
#  -fold '0,1,2,3,4' \
#  -dataset_dir '../dataset/twitter16' \
#  -batch_size 48 \
#  -savepath '../results/twitter16' \
#  -filter True \
#  -visible_gpu 0 \
#  -train_epoch 40 \
#  -log_tensorboard \
#  -warmup_steps 100 \
