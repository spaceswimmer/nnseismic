CUDA_VISIBLE_DEVICES=0 python infer.py --dataroot ../../data/synthetic_data/val  \
--sessions_path ../../data/synthetic_data/run/full_plus_rgt_4_Train/infer_50 \
--shape 128 128 128 --dataset_size 30 --num_workers 4 --only_load_input y \
--trained_model ../../data/synthetic_data/run/full_plus_rgt_4_Train/50.pth \
