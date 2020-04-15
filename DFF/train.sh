DATASET_ROOT_DIR='./kitti/'


CUDA_VISIBLE_DEVICES=0 nohup python train.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir "./kitti/" --dataset_split_file "./kitti/3DOP_splits/train.txt" >"./train.log" 2>&1 &
