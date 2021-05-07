import torch
from torch import nn
from math import log
# from src import utils
import argparse

def weights_init(module):

    classname = module.__class__.__name__

    if classname.find('Conv') != -1:
        module.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        module.weight.data.normal_(1.0, 0.02)
        module.bias.data.fill_(0)



class Discriminator(nn.Module):
    """Discriminator with input as features or image"""

    def __init__(self):
        super(Discriminator, self).__init__()

        # Store selfions for forward pass
        self.dis_input_sizes = [256, 128, 64]
        self.dis_output_sizes = [8, 4, 1]
        self.dis_input_num_channels = [196, 128, 256, 512, 512]
        self.dis_num_channels = 64
        self.dis_kernel_size = 4
        self.dis_kernel_size_io = 3
        self.dis_max_channels = 256

        # Calculate the amount of downsampling convolutional blocks
        num_down_blocks = int(log(
            self.dis_input_sizes[0] // max(self.dis_output_sizes[-1], 4), 2))

        # Read initial parameters
        in_channels = self.dis_input_num_channels[0]
        out_channels = self.dis_num_channels
        padding = (self.dis_kernel_size - 1) // 2
        padding_io = self.dis_kernel_size_io // 2
        spatial_size = self.dis_input_sizes[0]

        # Convolutional blocks
        self.blocks = nn.ModuleList()
        self.input_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()

        for i in range(num_down_blocks):

            # Downsampling block
            self.blocks += [nn.Sequential(
                nn.Conv2d(in_channels, out_channels, self.dis_kernel_size, 2,
                          padding),
                nn.LeakyReLU(0.2, True))]

            in_channels = out_channels
            spatial_size //= 2

            # If size of downsampling block's output is equal to one of the inputs
            if spatial_size in self.dis_input_sizes:

                # Get the number of channels in the next input
                in_channels = self.dis_input_num_channels[len(self.input_blocks)+1]

                self.input_blocks += [nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, self.dis_kernel_size_io, 1,
                              padding_io),
                    nn.LeakyReLU(0.2, True))]

                in_channels = out_channels * 2

            # If classifier is operating at the block's output size
            if spatial_size in self.dis_output_sizes:

                self.output_blocks += [
                    nn.Conv2d(in_channels, 1, self.dis_kernel_size_io, 1,
                              padding_io)]

            out_channels = min(out_channels * 2, self.dis_max_channels)

        # If 1x1 classifier is required at the end
        if self.dis_output_sizes[-1] == 1:
            self.output_blocks += [nn.Conv2d(out_channels, 2, 4, 4)]

        # Initialize weights
        self.apply(weights_init)

    def forward(self, img_dst, img_src=None, encoder=None):

        # Encode inputs
        # input = img_dst

        # Source input is concatenate via batch dim for speed up
        # if img_src is not None:
        #     input = torch.cat([input, img_src, 0])

        inputs = img_dst#encoder(input)

        # Reshape source inputs from batch dim to channels
        # if img_src is not None:
        #
        #     for i in range(len(inputs)):
        #         b, c, h, w = inputs[i].shape
        #         inputs[i] = inputs[i].view(b//2, c*2, h, w)

        output = inputs[0]
        # Current spatial size and indices for inputs and outputs
        spatial_size = output.shape[2]
        input_idx = 0
        output_idx = 0

        # List of multiscale predictions
        preds = []

        for block in self.blocks:

            output = block(output)

            spatial_size //= 2

            if spatial_size in self.dis_input_sizes:

                # Concatenate next input to current output
                input = self.input_blocks[input_idx](inputs[input_idx+1])

                output = torch.cat([output, input], 1)

                input_idx += 1

            if spatial_size in self.dis_output_sizes:

                # Predict probabilities in PatchGAN style
                preds += [self.output_blocks[output_idx](output)]

                output_idx += 1

        if 1 in self.dis_output_sizes:

            # Final probability prediction
            preds += [self.output_blocks[output_idx](output)]
        out_debug = preds[-1]
        out_debug = out_debug[:,:,0,0]#torch.reshape(out_debug, (-1, 1))
        return out_debug



