
CUDA_VISIBLE_DEVICES=2 nohup python run.py checkpoints/car_auto_T3_trainval/ --test --dataset_root_dir "./kitti/" --dataset_split_file "./kitti/3DOP_splits/val.txt" --output_dir "./inference/eval/" >"./run2.log" 2>&1 &

