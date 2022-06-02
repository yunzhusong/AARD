
python main.py \
  -early '0,6,12,18,24,30,36,42,48,54,60,120' \
  -fold '0' \
  -dataset_dir '../dataset/twitter16' \
  -batch_size 48 \
  -savepath '../results/twitter16/aard' \
  -filter True \
  -visible_gpu 0 \

python main.py \
  -early '0,6,12,18,24,30,36,42,48,54,60,120' \
  -fold '0,1,2,3,4' \
  -dataset_dir '../dataset/twitter15' \
  -batch_size 48 \
  -savepath '../results/twitter15/aard' \
  -filter True \
  -visible_gpu 0 \

python main.py \
 -early '0,60,120,240,480,720,1440,2880' \
 -fold '0,1,2,3,4' \
 -dataset_dir '../dataset/Pheme' \
 -batch_size 48 \
 -savepath '../results/Pheme/aard' \
 -filter True \
 -visible_gpu 0 \


