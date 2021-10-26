cd ..
CUDA_VISIBLE_DEVICES=6 python main.py --mode 'test' --mask_type 'radial' --sampling_rate 40\
 --batch_size 4 --test_path 'data/brain/db_valid/'