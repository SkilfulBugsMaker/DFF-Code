CUDA_VISIBLE_DEVICES=2 nohup python eval.py configs/car_auto_T3_train_eval_config --dataset_split_file "./splits/trainval_car.txt" >"./eval.log" 2>&1 &
