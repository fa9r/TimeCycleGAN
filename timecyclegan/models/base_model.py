"""
Defines an abstract model class
Class structure and naming convention are mostly similar to BaseModel of Pix2Pix.
"""

# pylint: disable=abstract-method

import os
from abc import ABC, abstractmethod
import itertools

import torch
import torchvision

from .model_utils import GANLoss, get_scheduler, count_params, set_trainable
from .networks import define_D, define_G


class BaseModel(ABC):
    """
    Abstract Model Class (Abstract Base Class)
    All subclasses have to overwrite __init__(), set_input(), forward(), and optimize_parameters().
    In __init__(), all subclasses have to set self.nets, self.net_names, self.loss_names and self.optimizers.
    In set_input(), all subclasses have to set self.real_source (and self.real_target if is_train=True).
    In forward(), all subclasses have to set self.fake_target (this is the generated image).
    In optimize_parameters(), the parameter update should be implemented.
    test() is a wrapper for forward() at inference time, where custom inference behaviour can be specified.
    """

    def __init__(
            self,
            is_train=True,
            gpu_ids=None,
            gan_mode='vanilla',
            generator_architecture='resnet_6blocks',
            discriminator_architecture='basic',
            width=256,
            height=256,
            batch_size=1,
            target_channels=3,
            generator_filters=64,
            discriminator_filters=64,
            learning_rate=0.0002,
            adam_beta1=0.5,
            init_epoch=0,
            num_epochs=200,
            global_step=0,
            block_size=1,
    ):
        """
        Initialize the model
        :param is_train: whether the model is called during train or test
        :param gpu_ids: list of GPU ids
        :param gan_mode: type of the gan loss ('vanilla', 'lsgan', 'wgangp')
        :param generator_architecture: architecture of the generator
            ('resnet_6blocks', 'resnet_9blocks', 'unet_256', 'unet_128')
        :param discriminator_architecture: architecture of the discriminator ('basic', 'n_layers', 'pixel')
        :param width: width of the source/target images
        :param height: height of the source/target images
        :param batch_size: batch size
        :param target_channels: number of channels in the target images
        :param generator_filters: number of filters in the last conv layer of the generator
        :param discriminator_filters: number of filters in the last conv layer of the discriminator
        :param learning_rate: initial learning rate
        :param adam_beta1: momentum term of adam optimizer
        :param init_epoch: start model training after init_epoch epochs (-> next training epoch is init_epoch+1)
        :param num_epochs: number of training epochs; this is needed to set niter and niter_decay in the schedulers
        :param global_step: global training step, used for logging
        """
        self.hparam_dict = {
            "GAN": gan_mode,
            "Generator": generator_architecture,
            "Discriminator": discriminator_architecture,
            "Width": width,
            "Height": height,
            "Generator Filters": generator_filters,
            "Discriminator Filters": discriminator_filters,
            "Adam Beta1": adam_beta1,
            "Block Size": block_size,
            "Learning Rate": learning_rate,
        }
        self.is_train = is_train
        self.gpu_ids = gpu_ids if gpu_ids is not None else []
        self.generator_architecture = generator_architecture
        self.discriminator_architecture = discriminator_architecture
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.target_channels = target_channels
        self.generator_filters = generator_filters
        self.discriminator_filters = discriminator_filters
        self.learning_rate = learning_rate
        self.adam_beta1 = adam_beta1
        self.init_epoch = init_epoch
        self.num_epochs = num_epochs
        self.step = global_step
        self.block_size = block_size
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.criterion_L1 = torch.nn.L1Loss()
        self.running_loss_step = 0
        self.criterion_GAN = GANLoss(gan_mode).to(self.device)
        self.loss_G, self.loss_D = [None] * 2  # init total G and D losses
        self.optimizer_G, self.optimizer_D, self.optimizers, self.schedulers = [None] * 4  # initialize optimizers
        self.tensorboard_writer, self.losses, self.images, self.running_losses = [None] * 4  # initialize logging args
        """!!! The following arguments have to be overwritten by all subclasses !!!"""
        self.generators, self.discriminators = [None] * 2  # all subclasses have to overwrite these in __init__()!
        self.real_target = None  # ground truth target; all subclasses have to overwrite this in set_input()!
        self.fake_target = None  # fake target; all subclasses have to overwrite this in forward()!


    @abstractmethod
    def set_input(self, input_data):
        """
        Unpack input data from data loader
        :param input_data: input from data loader
        """

    @abstractmethod
    def forward(self):
        """Forward pass"""

    @abstractmethod
    def backward_D(self):
        """Backward pass for the discriminator"""
        self.loss_D.backward()

    @abstractmethod
    def backward_G(self):
        """Backward pass for the generator"""
        self.loss_G.backward()

    def optimize_parameters(self):
        """Update model parameters"""
        self.loss_G, self.loss_D = [torch.tensor(0.0).to(self.device) for _ in range(2)]  # init loss_G and loss_D to zero
        self.forward()  # forward pass
        # update D
        set_trainable(self.discriminators.values(), True)  # enable backprop for D when optimizing D
        self.optimizer_D.zero_grad()  # set D's gradients to zero
        self.backward_D()  # calculate gradients for D
        self.optimizer_D.step()  # update D's weights
        # update G
        set_trainable(self.discriminators.values(), False)  # stop backprop for D when optimizing G
        self.optimizer_G.zero_grad()  # set generator gradients to zero
        self.backward_G()  # calculate gradients for G
        self.optimizer_G.step()  # update generator weights

    @abstractmethod
    def define_logging(self):
        """Define what should be logged to Tensorboard"""
        self.losses = {
            'Generator': self.loss_G,
            'Discriminator': self.loss_D,
        }
        self.images = {
            'Target': self.real_target,
            'Generations/Target': self.fake_target,
        }

    @abstractmethod
    def test(self):
        """
        Forward function used at test time
        :return: model prediction for current image
        """
        with torch.no_grad():
            self.forward()
        return self.fake_target

    def define_G(self, input_channels, output_channels, architecture=None, filters=None, **kwargs):
        """
        Define a generator.
        This is a wrapper for networks.define_G(), which sets ngf, netG, and gpu_ids based on class attributes
        :param input_channels: Number of input channels in the generator
        :param output_channels: Number of output channels in the generator
        :param architecture: Use an architecture other than self.generator_architecture
        :param filters: number of filters in the last conv layer of the generator
        :return: generator (torch.nn.Module)
        """
        return define_G(
            input_nc=input_channels,
            output_nc=output_channels,
            ngf=filters if filters is not None else self.generator_filters,
            netG=architecture if architecture is not None else self.generator_architecture,
            gpu_ids=self.gpu_ids,
            **kwargs,
        )

    def define_D(self, input_channels, architecture=None, filters=None, store_layers=False):
        """
        Define a discriminator.
        This is a wrapper for networks.define_D(), which sets ndf, netD, and gpu_ids based on class attributes
        :param input_channels: Number of input channels in the discriminator
        :param architecture: Use an architecture other than self.discriminator_architecture
        :param filters: number of filters in the last conv layer of the discriminator
        :param store_layers: If set, store list of individual discriminator layers (which we need for feature matching)
        :return: discriminator (torch.nn.Module)
        """
        return define_D(
            input_nc=input_channels,
            ndf=filters if filters is not None else self.discriminator_filters,
            netD=architecture if architecture is not None else self.discriminator_architecture,
            gpu_ids=self.gpu_ids,
            store_layers=store_layers,
        )

    def define_optimizer(self, params, chain=False):
        """
        Define an Adam optimizer
        This is a wrapper for torch.optim.Adam, which sets lr and betas based on class attributes
        :param params: parameters to be optimized
        :param chain: If set, treat params as a list of parameters and chain them together
        :return: Adam optimizer
        """
        params = itertools.chain(*params) if chain else params
        return torch.optim.Adam(params, lr=self.learning_rate, betas=(self.adam_beta1, 0.999))

    def init_optimizers(self):
        generator_params = [generator.parameters() for generator in self.generators.values()]
        discriminator_params = [discriminator.parameters() for discriminator in self.discriminators.values()]
        self.optimizer_G = self.define_optimizer(generator_params, chain=True)
        self.optimizer_D = self.define_optimizer(discriminator_params, chain=True)
        self.optimizers = [self.optimizer_G, self.optimizer_D]

    def init_schedulers(self):
        """Define schedulers for all optimizers and initialize them"""
        niter = self.num_epochs//2
        self.schedulers = [
            get_scheduler(optimizer, niter=niter, niter_decay=niter, epoch_count=self.init_epoch+1)
            for optimizer in self.optimizers
        ]

    def get_GAN_loss(self, images, discriminator, is_real=True, conditioned_on=None, store_features_in=None):
        """
        Wrapper of self.criterion_GAN with cleaner interface
        :param images: images (real/fake) to be discriminated
        :param discriminator: discriminator
        :param is_real: whether we want images to be classified as real or fake (i.e. False for fake images in D update)
        :param conditioned_on: additional discriminator inputs (will be concatenated to images first)
        :param store_features_in: if set, store pre-layer features in provided empty list (needed for feature matching)
        :return: discriminator output
        """
        if isinstance(images, list):
            images = torch.cat(images, 1)
        if isinstance(conditioned_on, list):
            conditioned_on = torch.cat(conditioned_on, 1)
        discriminator_input = images if conditioned_on is None else torch.cat((conditioned_on, images), 1)
        if store_features_in is not None:
            pred = discriminator(discriminator_input, store_features_in)
        else:
            pred = discriminator(discriminator_input)
        return self.criterion_GAN(pred, is_real)

    def get_GAN_loss_D(self, discriminator, real_images, fake_images, conditioned_on=None):
        if isinstance(fake_images, list):
            fake_images = torch.cat(fake_images, 1)  # concat lists of tensors so we can call .detach() later
        loss_real = self.get_GAN_loss(
            images=real_images,
            discriminator=discriminator,
            is_real=True,
            conditioned_on=conditioned_on,
        )
        loss_fake = self.get_GAN_loss(
            images=fake_images.detach(),
            discriminator=discriminator,
            is_real=False,
            conditioned_on=conditioned_on,
        )
        loss_total = (loss_real + loss_fake) / 2
        return loss_total, loss_real, loss_fake

    def update_learning_rate(self):
        """Update learning rates of all schedulers; called at the end of every epoch"""
        for scheduler in self.schedulers:
            scheduler.step()

    def save(self, save_dir, epoch=0):
        """
        Save all models
        :param save_dir: Directory to which the models are saved
        :param epoch: Current training epoch. This is used to have unique names during automatic checkpointing.
            If supplied, '_<epoch>' will be appended to all model names.
        """
        epoch_str = "_" + str(epoch) if epoch > 0 else ""
        for net_name, net in {**self.generators, **self.discriminators}.items():
            torch.save(net.state_dict(), os.path.join(save_dir, net_name + epoch_str + ".pth"))

    def load(self, load_dir, epoch=0):
        """
        Load all models
        We wrap this in a try-catch, because discriminators often become incompatible during spatio-temporal growing,
        in which case we just want to reinitialize them (by not loading anything) instead of aborting the training
        :param load_dir: Directory from which to load the models
        :param epoch: Training epoch after which the models were saved. This is used to load training checkpoints.
        """
        epoch_str = "_" + str(epoch) if epoch > 0 else ""
        for net_name, net in {**(self.generators or {}), **(self.discriminators or {})}.items():
            try:
                ckpt_path = os.path.join(load_dir, net_name + epoch_str + ".pth")
                state_dict = torch.load(ckpt_path, map_location=self.device)
                net.load_state_dict(state_dict)
                print("Succesfully loaded", net_name)
            except FileNotFoundError as exception:
                print("Error: Could not load", net_name, ". Error message:", exception)

    def print_params(self):
        """Print parameter count of all models"""
        print("------------------ Model Parameters --------------------")

        for net_name, net in {**self.generators, **self.discriminators}.items():
            params_total, params_trainable = count_params(net)
            print(net_name, "- total:", params_total, "| trainable:", params_trainable)
        print("--------------------------------------------------------")

    def init_losses(self):
        """
        Set all self.losses and self.running_losses to 0
        This is used in the training loop so that we can calculate running losses over multiple iterations
        """
        self.running_losses = {loss_name: 0 for loss_name in self.losses}
        self.running_loss_step = 0

    def update_step_and_losses(self):
        """
        Updates all self.running_losses (incrementing by the corresponding loss)
        This is used in the training loop so that we can calculate running losses over multiple iterations
        """
        self.define_logging()  # set losses and loss names
        if self.running_losses is None:
            self.init_losses()
        for loss_name, loss in self.losses.items():
            if loss:
                self.running_losses[loss_name] += float(loss)
        self.step += 1
        self.running_loss_step += 1

    def connect_tensorboard_writer(self, tensorboard_writer):
        """
        Connect to a tensorboard writer where all logs will be written
        :param tensorboard_writer: Tensoboard writer
        """
        self.tensorboard_writer = tensorboard_writer

    def log_hparams(self, additional_hparams=None):
        """
        Log all model hyperparameters to tensorboard
        :param additional_hparams: dict with additional hyperparameters to log (in addition to self.hparam_dict)
        :return: dict of all hyperparameters that were logged
        """
        additional_hparams = additional_hparams if additional_hparams is not None else {}
        hparams = {**self.hparam_dict, **additional_hparams}
        self.tensorboard_writer.add_hparams(hparams, {})
        return hparams

    def log_losses(self):
        """
        Log all self.running_losses to tensorboard
        :param log_every: logging frequency (i.e. over how many iterations we calculate the running losses)
        """
        for loss_name, running_loss in self.running_losses.items():
            self.tensorboard_writer.add_scalar('Loss/' + loss_name, running_loss / self.running_loss_step, self.step)
        self.init_losses()

    def log_image(self, image_name, image, nrow, channels, height, width, val_range=(-1, 1)):
        """
        Log a single image tensor to tensorboard
        :param image_name: Under which name to log the image tensor
        :param image: Image tensor to be logged
        :param nrow: How many images to log per row in the grid (cols will be set accordingly)
        :param channels: Number of channels of the individual images
        :param height: Image height
        :param width: Image width
        """
        image_flat = image.view((-1, channels, height, width))
        image_grid = torchvision.utils.make_grid(image_flat, nrow=(nrow if nrow > 1 else 8), normalize=True, range=val_range)
        self.tensorboard_writer.add_image(image_name, image_grid, self.step)

    def log_video(self, video_name, video, time, channels, height, width, val_range=(-1, 1), fps=10):
        video = video.view((-1, time, channels, height, width))
        video = (video - val_range[0]) / (val_range[1] - val_range[0])
        self.tensorboard_writer.add_video(tag=video_name, vid_tensor=video, global_step=self.step, fps=fps)

    def log_images(self):
        """
        Log all self.images to tensorboard
        self.images should be a dict {image_name: (image, image_data)},
            image_data is another dict containing non-default nrow/channel (e.g. set image_data['channels']=1 for B/W)
        """
        for image_name, image in self.images.items():
            image_data = {}
            # check if image data was provided
            if isinstance(image, (list, tuple)) and len(image) == 2 and isinstance(image[1], dict):
                image, image_data = image
            # if image is a list of tensors, concatenate them by channels
            if isinstance(image, list):
                image = torch.cat(image, 1)
            seq_len = image_data.get("nrow", self.block_size)
            channels = image_data.get("channels", self.target_channels)
            height = image_data.get("height", self.height)
            width = image_data.get("width", self.width)
            val_range = image_data.get("val_range", (-1, 1))
            self.log_image(image_name, image.detach(), seq_len, channels, height, width, val_range)
            if seq_len > 1 and channels == 3:
                self.log_video(image_name + "/video", image.detach(), seq_len, channels, height, width, val_range)

    def split_tensor(self, tensor, tensor_channels, chunks, dim=1):
        """
        Split a tensor-block by block size (into per-image tensors)
        This is basically a wrapper for torch.chunk which reshapes the result tensors
        :param tensor: tensor to be split
        :param tensor_channels: number of channels C
        :param chunks: number of chunks to split the tensor into
        :param dim: along which dim to split, must correspond to time axis, i.e. 1 for (N, T, C, H, W)
        :return: list of tensor parts
        """
        parts = torch.chunk(tensor, chunks=chunks, dim=dim)
        return [part.view(-1, tensor_channels, self.height, self.width) for part in parts]

    def get_placeholder_image(self, batch_size=None, channels=None, height=None, width=None, type=None):
        """
        Generate a placeholder image
        This is useful to provide recurrent generators with inputs for initial frames
        :param batch_size: batch size N
        :param channels: channels C
        :param height: height H
        :param width: width W
        :param type: placeholder type ('zero' | 'normal' | 'uniform'); default 'uniform'
        :return: random tensor (NxCxHxW)
        """
        batch_size = batch_size if batch_size is not None else self.batch_size
        channels = channels if channels is not None else self.target_channels
        height = height if height is not None else self.height
        width = width if width is not None else self.width
        type = type if type is not None else "normal"
        if type == "zero":
            init_function = torch.zeros  # 0
        elif type == "normal":
            init_function = torch.randn  # random N(0,1)
        elif type == "uniform":
            def init_function(size):
                """random U(-1,1)"""
                return torch.rand(size) * 2 - 1
        else:
            raise ValueError("Invalid argument: placeholder type %s not recognized." % type)
        return init_function((batch_size, channels, height, width)).to(self.device)

    def get_placeholder_images(self, num_images, batch_size=None, channels=None, height=None, width=None, type=None):
        """
        Generate a list of num_images random noise images (see get_random_target())
        """
        return [self.get_placeholder_image(batch_size, channels, height, width, type) for _ in range(num_images)]

    @staticmethod
    def print_data_statistics(tensor_dict):
        for tensor_name, tensor in tensor_dict.items():
            print(
                tensor_name, "- data stats:",
                "\n\tmin:", torch.min(tensor).item(),
                "\n\tmean:", torch.mean(tensor).item(),
                "\n\tmax:", torch.max(tensor).item()
            )


class ConditionalModel(BaseModel):
    """
    Abstract Base Class for conditional (image-to-image translation) models
    Defines some attributes that PairedModel and UnpairedModel have in common to avoid duplicate code
    """
    def __init__(self, source_channels=3, **kwargs):
        """
        Initialize model
        :param source_channels: number of channels in the source images
        """
        super().__init__(**kwargs)
        self.source_channels = source_channels
        self.real_source = None  # initialize ground truth source; all subclasses have to overwrite this in set_input()!

    def define_logging(self):
        super().define_logging()
        self.images['Source'] = (self.real_source, {'channels': self.source_channels})


class PairedModel(ConditionalModel):
    """
    Abstract Base Class for Paired Models
    Defines additional functions that are only used in paired training
    """
    def __init__(self, l1_loss_weight=0, feature_matching_loss_weight=0, perceptual_loss_weight=0, **kwargs):
        """
        Initialize the model
        :param l1_loss_weight: weight of the L1 loss during generator optimization
        :param feature_matching_loss_weight: relative weight of the feature matching loss
        :param perceptual_loss_weight: relative weight of the perceptual loss
        """
        from .networks.vgg19 import VGGLoss
        super().__init__(**kwargs)
        self.l1_loss_weight = l1_loss_weight
        self.hparam_dict["L1 Loss Weight"] = l1_loss_weight
        self.feature_matching_loss_weight = feature_matching_loss_weight
        self.hparam_dict["Feature Matching Loss Weight"] = feature_matching_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.hparam_dict["Perceptual Loss Weight"] = perceptual_loss_weight
        self.criterion_VGG = VGGLoss().to(self.device)

    def get_FM_loss(self, fake_images, real_images, discriminator, conditioned_on=None):
        """
        Calculate feature matching loss of a discriminator on given images
        :param fake_images: generated fake images
        :param real_images: corresponding real images
        :param discriminator: discriminator
        :param conditioned_on: additional discriminator inputs (will be concatenated to images first)
        :return: feature matching loss
        """
        feature_matching_loss = 0
        fake_features, real_features = [], []
        gan_loss = self.get_GAN_loss(
            images=fake_images,
            discriminator=discriminator,
            conditioned_on=conditioned_on,
            store_features_in=fake_features,
        )
        self.get_GAN_loss(
            images=real_images,
            discriminator=discriminator,
            conditioned_on=conditioned_on,
            store_features_in=real_features,
        )
        assert len(fake_features) == len(real_features)
        for fake_feature, real_feature in zip(fake_features, real_features):
            feature_matching_loss += self.criterion_L1(fake_feature, real_feature.detach())
        return gan_loss, feature_matching_loss


class UnpairedModel(ConditionalModel):
    """Abstract Base Class for Unpaired Models"""
    def __init__(self, cycle_loss_weight=0, source_to_target=True, **kwargs):
        """
        Initialize model
        :param cycle_loss_weight: relative weight of the cycle-consistency-loss
        :param source_to_target: during testing, whether we want to test source>target (True) or target>source (False)
        """
        super().__init__(**kwargs)
        self.hparam_dict["Cycle Loss Weight"] = cycle_loss_weight
        self.cycle_loss_weight = cycle_loss_weight
        self.source_to_target = source_to_target
        self.fake_source = None  # initialize fake source; all subclasses have to overwrite this in forward()!

    def define_logging(self):
        super().define_logging()
        self.images['Generations/Source'] = (self.fake_source, {'channels': self.source_channels})
