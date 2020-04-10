


CUDA_VISIBLE_DEVICES=2 nohup python evaluate.py --gpu 2 --model model_concat_upsa --dataset flying_things_dataset --data data_processed_maxcut_35_20k_2k_8192 --log_dir log_evaluate --model_path log_train/model.ckpt --num_point 2048 --batch_size 16 > "log_evaluate_flythings3d.txt" 2>&1 &
