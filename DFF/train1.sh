DATASET_ROOT_DIR='./kitti/'


CUDA_VISIBLE_DEVICES=1 nohup python train1.py configs/car_auto_T3_train_train_config configs/car_auto_T3_train_config --dataset_root_dir "./kitti/" --dataset_split_file "./kitti/3DOP_splits/train.txt" >"./train1.log" 2>&1 &
