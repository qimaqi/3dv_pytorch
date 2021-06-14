# Image reconstruction with R2D2


## Environment Preparation 

``` 
> pip install -r requirements.txt

```
## dataset preparation 
Download Megadepth and NYU dataset at [dgrive](https://drive.google.com/file/d/1StpUiEauckZcxHZeBzoq6L2K7pcB9v3E/view?usp=drive_open). Untar the file at the project folder. Please change the path to your own directory.

And run R2D2_data_preparation.py to prepare R2D2 data.

## Training Reconstruction Network
Use train_coarse.py to use the reconstruction network. Before running, please download vgg16 pretrained model from https://drive.google.com/drive/folders/17WY-RxN3G3uLBclI_wvftXMQZWIwd6q8?usp=sharing.  

The weight are set to be 1, 5 for pixel loss and perceptron loss respectively, trained with 24 epochs. The images are reconstructed with high quality. The sparsity of input feature could be manipulated in the corresponding dataloader. Please change dir_checkpoint to save the pretrained model to your local directory. The currently used network is Unet, you can change to InvNet or UNet_Nested according to your demand.

