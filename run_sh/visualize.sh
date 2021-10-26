# mask_type can be 'radial', 'cartes', 'guass', 'spiral'
cd ..
CUDA_VISIBLE_DEVICES=0 python main.py --mode 'visualize' --mask_type 'radial' --test_path 'data/brain/db_valid/'