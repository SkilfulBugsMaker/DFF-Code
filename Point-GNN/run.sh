
CUDA_VISIBLE_DEVICES=2 nohup python run.py checkpoints/car_auto_T3_train/ --test --dataset_root_dir "./kitti/" --output_dir "inference/" >"./run.log" 2>&1 &

