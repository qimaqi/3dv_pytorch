# Image reconstruction with SuperPoint and R2D2


## Environment Preparation 

``` 
> pip install -r requirements.txt

```
## dataset preparation 
Download Megadepth and NYU dataset at [dgrive](https://drive.google.com/file/d/1StpUiEauckZcxHZeBzoq6L2K7pcB9v3E/view?usp=drive_open). Untar the file at the project folder.

Dwonload SuperPoint Pretrained Weight from [repo](https://github.com/magicleap/SuperPointPretrainedNetwork.git). And run Superpoint_data_preparation.py to prepare SuperPoint data.

## Training Reconstruction Network
Use train_coarse.py to use the reconstruction network.

> You should prepare some necessary pth file.
>
> 1. download vgg16-397923af.pth from [data](https://drive.google.com/drive/folders/17WY-RxN3G3uLBclI_wvftXMQZWIwd6q8?usp=sharing) and put in same working path
> 2. download  superpoint_v1.pth [repo](https://github.com/magicleap/SuperPointPretrainedNetwork.git) in folder utils/ to make path './utils/superpoint_v1.pth'

The weight are set to be 1, 5 for pixel loss and perceptron loss respectively, trained with 24 epochs. The images are reconstructed with high quality. The spacity of input feature could be manupulated in the corresponding dataloader.

> Hint: some configuration you could make inside train_coarse.py
>
> 1. Different network backbone
>
>    ```python
>    ########### change here if you want to change different model ##############
>    # net = InvNet(n_channels=256, n_classes=1)    
>    # net = UNet_Nested(n_channels=input_channel, n_classes=output_channel)
>    net = UNet(n_channels=input_channel, n_classes=output_channel, bilinear=True)
>    ```
>
> 2. Different data processing strategy
>
>    ```python
>        ########### if you want to use already generate feature then use offline below ##############
>        dataset = dataset_superpoint_5k(image_list,feature_list,img_scale, crop_size, max_points)
>        val_dataset = dataset_superpoint_5k(val_image_list,val_feature_list,img_scale, crop_size, max_points)
>        
>        ############## if you want to use superpoint online in parallel to process with data ##############
>        # dataset = dataset_superpoint_5k_online(image_list,feature_list,img_scale, pct_3D_points, crop_size, max_points)  
>        # val_dataset = dataset_superpoint_5k_online(val_image_list,val_feature_list,img_scale, pct_3D_points, crop_size, max_points)
>    ```
>
> 3. Different optimizer and learning rate scheduler strategies
>
>    ```python
>        ########## different optimizer and learning rate scheduler strategy ##############
>        #optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
>        optimizer = optim.Adam(net.parameters(), lr=lr, eps = 1e-8)
>        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2)
>        #scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=1) pytorch 1.01
>        #scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5,12,16], gamma=0.1)
>    ```
>
>    

## Training Refine Network

Use train_refine.py to use the refine net model that was deployed on training data.

The weight are set to be 1, 5, 10 for pixel loss, perceptron loss and discriminator loss respectively, trained with 24 epochs. It was not adopted in final roadmap because it failed to provide realistic color reconstruction information. Note that in the origina paper, the network takes an extra channel of depth, which would act as semi-segmentation inforamtion for network. 



## Result

64 Epochs are trained and result is shown below:

- process results

  ![...](https://github.com/AlexanderNevada/3dv_pytorch/blob/main/readme_image/process_result.png)  

* Good results  
  ![...](https://github.com/AlexanderNevada/3dv_pytorch/blob/main/readme_image/indoor_reconstruction.png)  

  ![...](https://github.com/AlexanderNevada/3dv_pytorch/blob/main/readme_image/face_reconstruction.png)  

  ![...](https://github.com/AlexanderNevada/3dv_pytorch/blob/main/readme_image/text_reconstruction.png)

  
