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

The weight are set to be 1, 5 for pixel loss and perceptron loss respectively, trained with 24 epochs. The images are reconstructed with high quality. The spacity of input feature could be manupulated in the corresponding dataloader.

## Training Refine Network

Use train_refine.py to use the refine net model that was deployed on training data.

The weight are set to be 1, 5, 10 for pixel loss, perceptron loss and discriminator loss respectively, trained with 24 epochs. It was not adopted in final roadmap because it failed to provide realistic color reconstruction information. Note that in the origina paper, the network takes an extra channel of depth, which would act as semi-segmentation inforamtion for network. 
