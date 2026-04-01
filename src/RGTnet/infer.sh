CUDA_VISIBLE_DEVICES=3 python infer.py --dataroot datasets/syn  \
--shape 256 256 128 --dataset_size 8 --num_workers 4 --only_load_input y \
--trained_model checkpoints/trained_RGTNet.pth
