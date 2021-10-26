# mask_type can be 'radial', 'cartes', 'guass', 'spiral'
# sampling_rate can be 10, 20, 30, 40, 50, 60, 70, 80, 90
cd ..
CUDA_VISIBLE_DEVICES=0 python main.py --mode 'test' --mask_type 'radial' --sampling_rate 10\
 --batch_size 4 --test_path 'data/brain/db_valid/'