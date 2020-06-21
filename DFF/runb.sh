
CUDA_VISIBLE_DEVICES=2 nohup python run-boxlevel.py checkpoints/car_auto_T3_train/ --dataset_root_dir "./kitti/" --dataset_split_file "./kitti/3DOP_splits/val.txt" --output_dir "./inference/" >"./runb.log" 2>&1 &

