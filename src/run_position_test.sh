python main.py \
  -test_gen \
  -run_exp \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/twitter16' \
  -batch_size 1 \
  -savepath '../results/twitter16' \
  -visible_gpu 0 \
  -log_tensorboard \

