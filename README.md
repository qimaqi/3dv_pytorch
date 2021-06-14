# Image reconstruction with SuperPoint and R2D2


## Environment Preparation 

``` 
> pip install -r requirements.txt

```
## dataset preparation 
[Superpoint pretrained Weight](https://github.com/magicleap/SuperPointPretrainedNetwork.git)

## Training Reconstruction Network


## Training Refine Network

Use train_refine.py to use the refine net model that was deployed on training data.

The weight are set to be 1, 5, 10 for pixel loss, perceptron loss and discriminator loss respectively, trained with 24 epochs. It was not adopted in final roadmap because it failed to provide realistic color reconstruction information. Note that in the origina paper, the network takes an extra channel of depth, which would act as semi-segmentation inforamtion for network. 
