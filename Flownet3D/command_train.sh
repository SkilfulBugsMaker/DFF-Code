LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
PATH=$PATH::/usr/local/cuda-10.0/bin
CUDA_HOME=/usr/local/cuda-10.0


CUDA_VISIBLE_DEVICES=2 LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64 PATH=$PATH::/usr/local/cuda-10.0/bin CUDA_HOME=/usr/local/cuda-10.0 nohup python train.py --gpu 2 --model model_concat_upsa --data data_processed_maxcut_35_20k_2k_8192 --log_dir log_train --num_point 2048 --max_epoch 151 --learning_rate 0.001 --batch_size 8 >"./log_train.txt" 2>&1 &
