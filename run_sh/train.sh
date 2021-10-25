cd ..
CUDA_VISIBLE_DEVICES=0 python main.py --mode 'train' --mask_type 'radial' --sampling_rate 10\
 --batch_size 4 --num_epoch 500 --train_path 'data/brain/db_train/' --val_path 'data/brain/db_valid/'
