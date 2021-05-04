# InvNet: generate image from feature



### Training

```shell script
> python train.py -h
usage: train.py [-h] [-e E] [-b [B]] [-l [LR]] [-f LOAD] [-s SCALE] [-v VAL]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  -e E, --epochs E      Number of epochs (default: 5)
  -b [B], --batch-size [B]
                        Batch size (default: 1)
  -l [LR], --learning-rate [LR]
                        Learning rate (default: 0.1)
  -f LOAD, --load LOAD  Load model from a .pth file (default: False)
  -s SCALE, --scale SCALE
                        Downscaling factor of the images (default: 0.5)
  -v VAL, --validation VAL
                        Percent of the data that is used as validation (0-100)
                        (default: 15.0)

```
By default, the `scale` is 0.5, so if you wish to obtain better results (but use more memory), set it to 1.

The input images and target masks should be in the `data/imgs` and `data/masks` folders respectively.

### Pretrained model
A [pretrained model](https://github.com/milesial/Pytorch-UNet/releases/tag/v1.0) is available for the Carvana dataset. It can also be loaded from torch.hub:

```python
net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')
```
The training was done with a 100% scale and bilinear upsampling.

## Tensorboard
You can visualize in real time the train and test losses, the weights and gradients, along with the model predictions with tensorboard:

`tensorboard --logdir=runs`

You can find a reference training run with the Caravana dataset on [TensorBoard.dev](https://tensorboard.dev/experiment/1m1Ql50MSJixCbG1m9EcDQ/#scalars&_smoothingWeight=0.6) (only scalars are shown currently).

## Notes on memory

The model has be trained from scratch on a GTX970M 3GB.
Predicting images of 1918*1280 takes 1.5GB of memory.
Training takes much approximately 3GB, so if you are a few MB shy of memory, consider turning off all graphical displays.
This assumes you use bilinear up-sampling, and not transposed convolution in the model.

## Support

Personalized support for issues with this repository, or integrating with your own dataset, available on [xs:code](https://xscode.com/milesial/Pytorch-UNet).


---

Original paper by Olaf Ronneberger, Philipp Fischer, Thomas Brox: [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

![network architecture](https://i.imgur.com/jeDVpqF.png)
