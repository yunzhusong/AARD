
python main.py \
  -train_detector \
  -quat '5,10,25,50,75,100' \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/twitter16' \
  -batch_size 48 \
  -savepath '../results/twitter16/aard' \
  -filter True \
  -visible_gpu 0 \
  -log_tensorboard \

#python main.py \
#  -train_detector \
#  -quat '5,10,25,50,75,100' \
#  -fold '0,1,2,3,4' \
#  -dataset_dir '../dataset/twitter15' \
#  -batch_size 48 \
#  -savepath '../results/twitter15/aard' \
#  -filter True \
#  -visible_gpu 0 \
#  -log_tensorboard \
#
#python main.py \
#  -train_detector \
#  -quat '5,10,25,50,75,100' \
#  -fold '0,1,2,3,4' \
#  -dataset_dir '../dataset/Pheme' \
#  -batch_size 48 \
#  -savepath '../results/pheme/aard' \
#  -filter True \
#  -visible_gpu 0 \
#  -log_tensorboard \
#
