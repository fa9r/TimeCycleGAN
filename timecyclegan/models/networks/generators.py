"""
Generator Definitions
Based on networks.py from official PyTorch Pix2Pix repo.
"""
# pylint: disable-all

import functools

import torch
import torch.nn as nn

from .network_utils import get_norm_layer, init_net, get_activation_function


def define_G(input_nc, output_nc, ngf, netG, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], output_activation='tanh', part=None):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer
        netG (str) -- the architecture's name: resnet_9blocks | resnet_6blocks | unet_256 | unet_128
        norm (str) -- the name of normalization layers used in the network: batch | instance | none
        use_dropout (bool) -- if use dropout layers.
        init_type (str)    -- the name of our initialization method.
        init_gain (float)  -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
        output_activation (str) -- output activation of the generator (tanh | sigmoid | ...)
        part (str) -- if given, only return a part of the generator: down | up | head | up_head

    Returns a generator

    Our current implementation provides two types of generators:
        U-Net: [unet_128] (for 128x128 input images) and [unet_256] (for 256x256 input images)
        The original U-Net paper: https://arxiv.org/abs/1505.04597

        Resnet-based generator: [resnet_6blocks] (with 6 Resnet blocks) and [resnet_9blocks] (with 9 Resnet blocks)
        Resnet-based generator consists of several Resnet blocks between a few downsampling/upsampling operations.
        We adapt Torch code from Justin Johnson's neural style transfer project (https://github.com/jcjohnson/fast-neural-style).


    The generator has been initialized by <init_net>. It uses RELU for non-linearity.
    """
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    output_fn = get_activation_function(activation_type=output_activation)

    if netG == 'dcgan':
        net = DCGANGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer)
    elif netG == 'dcgan_enc':
        net = DCGANEncoder(input_nc, output_nc, ngf, norm_layer=norm_layer)
    elif netG.startswith('resnet'):
        n_blocks = int(netG[7:-6])
        if part == "down":
            net = ResnetFeatureExtractor(
                input_nc=input_nc,
                ngf=ngf,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                n_blocks=n_blocks,
            )
        elif part == "up":
            net = ResnetUpsampler(
                ngf=ngf,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                n_blocks=n_blocks,
            )
        elif part == "head":
            net = ResnetOutputHead(
                output_nc=output_nc,
                ngf=ngf,
                output_fn=output_fn,
            )
        elif part == "down_up":
            net = ResnetGeneratorWithoutHead(
                input_nc=input_nc,
                ngf=ngf,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                n_blocks=n_blocks
            )
        elif part == "up_head":
            net = ResnetGeneratorFromFeatures(
                output_nc=output_nc,
                ngf=ngf,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                n_blocks=n_blocks,
                output_fn=output_fn
            )
        else:
            net = ResnetGenerator(
                input_nc=input_nc,
                output_nc=output_nc,
                ngf=ngf,
                norm_layer=norm_layer,
                use_dropout=use_dropout,
                n_blocks=n_blocks,
                output_fn=output_fn
            )
    elif netG == 'unet_128':
        net = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, output_fn=output_fn)
    elif netG == 'unet_256':
        net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, output_fn=output_fn)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


class DCGANEncoder(nn.Module):
    """
    DCGAN encoder to encode 64 x 64 images to enc_nc x 1 x 1
    Used to encode previous images in sequential generation
    Effectively a reversed version of the normal DCGANGenerator
    """
    def __init__(self, input_nc=3, enc_nc=100, ngf=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.main = nn.Sequential(
            nn.Conv2d(input_nc, ngf, 4, 2, 1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
            # state size: ngf x 32 x 32
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            # state size: ngf*2 x 16 x 16
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
            # state size: ngf*4 x 8 x 8
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 8),
            nn.ReLU(True),
            # state size: ngf*8 x 4 x 4
            nn.Conv2d(ngf * 8, enc_nc, 4, 1, 0, bias=use_bias),
            #norm_layer(enc_nc),
            #nn.ReLU(True),
            # state size: enc_nc x 1 x 1
        )

    def forward(self, input):
        return self.main(input)


class DCGANGenerator(nn.Module):
    """
    DCGAN generator for 64x64 images
    Adjusted from official PyTorch tutorial https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
    """
    def __init__(self, input_nc=100, output_nc=3, ngf=64, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(input_nc, ngf * 8, 4, 1, 0, bias=use_bias),
            norm_layer(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=use_bias),
            norm_layer(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, output_nc, 4, 2, 1, bias=use_bias),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


def define_resnet_feature_extractor(input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
    """Construct a resnet-based feature extractor - first half of ResnetGenerator"""
    assert (n_blocks >= 0)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    model = [nn.ReflectionPad2d(3),
             nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
             norm_layer(ngf),
             nn.ReLU(True)]

    n_downsampling = 2
    for i in range(n_downsampling):  # add downsampling layers
        mult = 2 ** i
        model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                  norm_layer(ngf * mult * 2),
                  nn.ReLU(True)]

    mult = 2 ** n_downsampling
    for i in range(n_blocks):  # add ResNet blocks
        model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                              use_bias=use_bias)]
    return model


def define_resnet_feature_upsampler(ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
    assert (n_blocks >= 0)
    if type(norm_layer) == functools.partial:
        use_bias = norm_layer.func == nn.InstanceNorm2d
    else:
        use_bias = norm_layer == nn.InstanceNorm2d

    model = []
    n_downsampling = 2
    mult = 2 ** n_downsampling
    for i in range(n_blocks):  # add ResNet blocks
        model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                              use_bias=use_bias)]

    for i in range(n_downsampling):  # add upsampling layers
        mult = 2 ** (n_downsampling - i)
        model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                     kernel_size=3, stride=2,
                                     padding=1, output_padding=1,
                                     bias=use_bias),
                  norm_layer(int(ngf * mult / 2)),
                  nn.ReLU(True)]
    return model


def define_resnet_output_head(output_nc, ngf=64, output_fn=nn.Tanh()):
    return [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), output_fn]


class ResnetPart(nn.Module):
    def forward(self, input):
        return self.model(input)


class ResnetFeatureExtractor(ResnetPart):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super().__init__()
        model = define_resnet_feature_extractor(
            input_nc=input_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks,
            padding_type=padding_type
        )
        self.model = nn.Sequential(*model)


class ResnetUpsampler(ResnetPart):
    def __init__(self, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super().__init__()
        model = define_resnet_feature_upsampler(
            ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, padding_type=padding_type
        )
        self.model = nn.Sequential(*model)


class ResnetOutputHead(ResnetPart):
    def __init__(self, output_nc, ngf=64, output_fn=nn.Tanh()):
        super().__init__()
        model = define_resnet_output_head(output_nc=output_nc, ngf=ngf, output_fn=output_fn)
        self.model = nn.Sequential(*model)


class ResnetGeneratorFromFeatures(ResnetPart):
    def __init__(self, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', output_fn=nn.Tanh()):
        super().__init__()
        upsampler = define_resnet_feature_upsampler(
            ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks, padding_type=padding_type
        )
        output_head = define_resnet_output_head(output_nc=output_nc, ngf=ngf, output_fn=output_fn)
        self.model = nn.Sequential(*upsampler, *output_head)


class ResnetGeneratorWithoutHead(ResnetPart):
    def __init__(self, input_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super().__init__()
        n_blocks_down = n_blocks // 2
        n_blocks_up = n_blocks - n_blocks_down
        downsampler = define_resnet_feature_extractor(
            input_nc=input_nc, ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks_down,
            padding_type=padding_type
        )
        upsampler = define_resnet_feature_upsampler(
            ngf=ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=n_blocks_up, padding_type=padding_type
        )
        self.model = nn.Sequential(*downsampler, *upsampler)


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', output_fn=nn.Tanh()):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [output_fn]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, output_fn=nn.Tanh()):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer, output_fn=output_fn)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, output_fn=nn.Tanh()):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            user_dropout (bool) -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, output_fn]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)
