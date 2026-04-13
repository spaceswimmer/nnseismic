CUDA_VISIBLE_DEVICES=0 python infer.py --dataroot /mnt/storage/nnseismic/synthetic_data/val  \
--sessions_path /mnt/storage/nnseismic/runs/full_rgt_1_Train \
--shape 128 128 128 --dataset_size 8 --num_workers 4 --only_load_input y \
--trained_model /mnt/storage/nnseismic/runs/full_rgt_1_Train/checkpoint/best_model.pth \
--dataset_size 30
