## RefineGAN
unofficial pytorch implementation of RefineGAN (https://arxiv.org/abs/1709.00753) for CSMRI reconstruction, the official code using tensorpack can be found at https://github.com/tmquan/RefineGAN

### To Do
- [ ] run the original tensorpack code (sorry, can't run tensorpack on my GPU)
- [x] pytorch implementation and experiments on brain images with radial mask
- [x] bug fixed. ~~the mean psnr of zero-filled image is not exactly the same as the value in original paper, although the model improvement is similar~~
- [ ] experiments on different masks

### Install
python>=3.7.11 is required with all requirements.txt installed including pytorch>=1.10.0
```shell
git clone https://github.com/hellopipu/RefineGAN.git
cd RefineGAN
pip install -r requirements.txt
```
### How to use
#### **for training:**
```shell
cd run_sh
sh train.sh
```
the model will be saved in folder `weight`, tensorboard information will be saved in folder `log`. You can change the arguments in script such as `--mask_type` and `--sampling_rate` for different experiment settings.


#### **for tensorboard:**

check the training curves while training
```shell
tensorboard --logdir log
```
the training info of my experiments is already in `log` folder

#### **for testing:**

test after training, or you can download my trained model weights from [google drive](https://drive.google.com/drive/folders/1ZSA_rOy9OCRfRjpX35gOPK8WiDCRFAn8?usp=sharing).

```shell
cd run_sh
sh test.sh
```

#### **for visualization:**
```shell
cd run_sh
sh visualize.sh
```



### training curves
sampling rates : 10%(light orange), 20%(dark blue), 30%(dark orange), 40%(light blue). You can check more loss curves of my experiments using tensorboard.

| loss_G_loss_total    | loss_recon_img_Aa     |
|------------|-------------|
|<img src="img/loss_G_loss_total.svg?raw=true" title = "G_loss_total" width="400">|<img src="img/loss_recon_img_Aa.svg?raw=true" title="loss_recon_img_Aa" width="400">|

PSNR on training set over 500 epochs, compared with results shown in original paper. 

| my_train_psnr    | paper_train_psnr    |
|------------|-------------|
|<img src="img/my_train_psnr.svg?raw=true" title = "my_train_psnr" width="400">|<img src="img/paper_train_psnr.png?raw=true" title="paper_train_psnr" width="400">|

### Test results

mean PSNR on validation dataset with radial mask of different sampling rates, batch_size is set as 4;

model  |  10%  | 20%  | 30% | 40% 
-------|-------|------|-----|------
zero-filled| 22.296 | 25.806 | 28.997| 31.699 
RefineGAN|  32.705 |  36.734 | 39.961| 42.903

### Test cases visualization
<<<<<<< HEAD
| sampling rate    | from left to right: mask, zero-filled, prediction and ground truth | error (zero-filled) and error (prediction)    
-------|------|------
10%|<img src="img/radial_10.png?raw=true" title = "radial_10" width="600">|<img src="img/radial_10_error.png?raw=true" title = "radial_10_error" width="400">
20%|<img src="img/radial_20.png?raw=true" title = "radial_20" width="600">|<img src="img/radial_20_error.png?raw=true" title = "radial_20_error" width="400">
30%|<img src="img/radial_30.png?raw=true" title = "radial_30" width="600">|<img src="img/radial_30_error.png?raw=true" title = "radial_30_error" width="400">
40%|<img src="img/radial_40.png?raw=true" title = "radial_40" width="600">|<img src="img/radial_40_error.png?raw=true" title = "radial_40_error" width="400">
=======
| rate    | ------ mask -------------- zero-filled ----------- prediction ---- ground truth - error (zero-filled) - error (prediction)    
-------|------
10%|<img src="img/radial_10.png?raw=true" title = "radial_10" width="800"> 
20%|<img src="img/radial_20.png?raw=true" title = "radial_20" width="800"> 
30%|<img src="img/radial_30.png?raw=true" title = "radial_30" width="800"> 
40%|<img src="img/radial_40.png?raw=true" title = "radial_40" width="800"> 

>>>>>>> ded67b17301e7bde068fb737f9ac6231d8e7885b
### Notes on RefineGAN

- data processing before training : complex value represents in 2-channel , each channel rescale to [-1,1]; accordingly the last layer of generator is tanh()
- Generator uses residual learning for reconstruction task
- Generator is a cascade of two U-net, the U-net doesn't do concatenation but addition when combining the enc and dec features. 
- each U-net is followed by a Data-consistency (DC) module, although the paper doesn't mention it.
- the last layer of generator is tanh layer on two-channel output, so when we revert output to original pixel scale and
calculate abs, the pixel value may exceed 255; we need to do clipping while calculating psnr
- while training, we get two random image samples A, B for each iteration, RefineGAN calculates **a large amount of losses** (it may be redundant)
including reconstruction loss on different phases of generator output in both image domain and frequency domain, total
variantion loss and WGAN loss
- one special loss is D_loss_AB, D is trained to only distinguish from real samples and fake samples, so D should not only work for (real A, fake A) or (real B, fake B), but also work for (real A, fake B) input
- WGAN-gp may be used to improve the performance
- small batch size MAY BE better. In my experiment, batch_size=4 is better than batch_size=16

**_I will appreciate if you can find any implementation mistakes in codes._**


