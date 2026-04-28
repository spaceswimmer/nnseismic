CUDA_VISIBLE_DEVICES=0 python infer.py --dataroot /mnt/storage/nnseismic/real_data  \
--sessions_path /mnt/storage/nnseismic/runs/full_plus_rgt_3_Infer \
--shape 128 128 128 --dataset_size 30 --num_workers 4 --only_load_input y \
--trained_model /mnt/storage/nnseismic/runs/full_plus_rgt_3_Train/checkpoint/70.pth \
