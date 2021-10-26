cd ..
CUDA_VISIBLE_DEVICES=6 python main.py --mode 'train' --mask_type 'radial' --sampling_rate 40\
 --batch_size 4 --num_epoch 500 --train_path 'data/brain/db_train/' --val_path 'data/brain/db_train/'
