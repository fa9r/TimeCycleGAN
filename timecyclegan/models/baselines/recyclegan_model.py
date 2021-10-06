"""Reimplemented RecycleGAN (https://github.com/aayushbansal/Recycle-GAN/) model"""

import torch

from timecyclegan.models.base_model import UnpairedModel, set_trainable
from timecyclegan.models.model_utils import ImagePool


class RecycleGANModel(UnpairedModel):
    """RecycleGAN Model"""

    def __init__(self, fix_GAN_loss_2r=False, **kwargs):
        super().__init__(**kwargs)
        self.fix_GAN_loss_2r = fix_GAN_loss_2r
        self.identity_loss_weight = 0  # don't use identity loss, as done in original RecycleGAN
        self.pool_size = 0  # don't use pool, as done in original RecycleGAN
        self.p_architecture = "unet_256" if self.width == 256 else "resnet_6blocks"
        # Define generators
        self.generator_target = self.define_G(self.source_channels, self.target_channels)
        self.generator_source = self.define_G(self.target_channels, self.source_channels)
        self.generators = {
            "generator_target": self.generator_target,
            "generator_source": self.generator_source,
        }
        if self.is_train:
            # Define discriminators
            self.discriminator_target = self.define_D(self.target_channels)
            self.discriminator_source = self.define_D(self.source_channels)
            self.discriminators = {
                "discriminator_target": self.discriminator_target,
                "discriminator_source": self.discriminator_source,
            }
            # define predictors (motion models)
            self.predictor_target = self.define_G(self.target_channels*2, self.target_channels, self.p_architecture)
            self.predictor_source = self.define_G(self.source_channels*2, self.source_channels, self.p_architecture)
            self.generators["predictor_target"] = self.predictor_target
            self.generators["predictor_source"] = self.predictor_source
            # initialize image buffers to store previously generated images for discriminator training
            self.fake_source_pool, self.fake_target_pool = ImagePool(self.pool_size), ImagePool(self.pool_size)
            # initialize source and target parts which will be set in set_input() during training
            self.real_source_0, self.real_source_1, self.real_source_2 = [None] * 3
            self.real_target_0, self.real_target_1, self.real_target_2 = [None] * 3
            # Define other generations that will be set in forward() during training
            self.fake_target_0, self.fake_target_1, self.fake_target_2f = [None] * 3
            self.fake_source_0, self.fake_source_1, self.fake_source_2f, self.fake_source = [None] * 4
            self.fake_source_2r, self.fake_target_2r, self.fake_fake_source_2, self.fake_fake_target_2 = [None] * 4
            # Define losses that will be set in optimize_parameters() during training
            self.loss_G_target_GAN_0, self.loss_G_target_GAN_1 = [None] * 2
            self.loss_G_target_GAN_2f, self.loss_G_target_GAN_2r = [None] * 2
            self.loss_G_source_GAN_0, self.loss_G_source_GAN_1 = [None] * 2
            self.loss_G_source_GAN_2f, self.loss_G_source_GAN_2r = [None] * 2
            self.loss_G_cycle_target, self.loss_G_cycle_source = [None] * 2
            self.loss_G_cycle_target_time, self.loss_G_cycle_source_time = [None] * 2
            self.loss_D_target, self.loss_D_source = [None] * 2
            self.loss_D_target_real_0, self.loss_D_target_real_1, self.loss_D_target_real_2 = [None] * 3
            self.loss_D_target_fake_0, self.loss_D_target_fake_1 = [None] * 2
            self.loss_D_target_fake_2f, self.loss_D_target_fake_2r = [None] * 2
            self.loss_D_source_real_0, self.loss_D_source_real_1, self.loss_D_source_real_2 = [None] * 3
            self.loss_D_source_fake_0, self.loss_D_source_fake_1 = [None] * 2
            self.loss_D_source_fake_2f, self.loss_D_source_fake_2r = [None] * 2

    def set_input(self, input_data):
        if self.is_train:
            source, target = input_data
            self.real_source = source["source"].to(self.device)
            self.real_source = self.split_tensor(self.real_source, self.source_channels, 3)
            self.real_source_0, self.real_source_1, self.real_source_2 = self.real_source
            self.real_target = target["target"].to(self.device)
            self.real_target = self.split_tensor(self.real_target, self.target_channels, 3)
            self.real_target_0, self.real_target_1, self.real_target_2 = self.real_target
        elif self.source_to_target:
            self.real_source = input_data["source"].to(self.device)
        else:
            self.real_target = input_data["target"].to(self.device)

    def forward(self):
        # fake target images 0 and 1, predicted by target generator
        self.fake_target_0 = self.generator_target(self.real_source_0)
        self.fake_target_1 = self.generator_target(self.real_source_1)
        # predict two fakes for target image 2, one from real target images, one from fakes
        self.fake_target_2f = self.predictor_target(torch.cat((self.fake_target_0, self.fake_target_1), 1))
        self.fake_target_2r = self.predictor_target(torch.cat((self.real_target_0, self.real_target_1), 1))
        # source target images 0 and 1, predicted by source generator
        self.fake_source_0 = self.generator_source(self.real_target_0)
        self.fake_source_1 = self.generator_source(self.real_target_1)
        # predict two fakes for source image 2, one from real source images, one from fakes
        self.fake_source_2f = self.predictor_source(torch.cat((self.fake_source_0, self.fake_source_1), 1))
        self.fake_source_2r = self.predictor_source(torch.cat((self.real_source_0, self.real_source_1), 1))
        # project fakes of image 2 (generated from other fakes) back for cycle loss
        self.fake_fake_source_2 = self.generator_source(self.fake_target_2f)
        self.fake_fake_target_2 = self.generator_target(self.fake_source_2f)
        # concatenate fake targets for logging
        self.fake_target = [self.fake_target_0, self.fake_target_1, self.fake_target_2f]
        # concatenate fake sources for logging
        self.fake_source = [self.fake_source_0, self.fake_source_1, self.fake_source_2f]

    def backward_G(self):
        """Calculate generator losses"""
        # GAN loss of target generator
        self.loss_G_target_GAN_0 = self.get_GAN_loss(self.fake_target_0, self.discriminator_target)
        self.loss_G_target_GAN_1 = self.get_GAN_loss(self.fake_target_1, self.discriminator_target)
        self.loss_G_target_GAN_2f = self.get_GAN_loss(self.fake_target_2f, self.discriminator_target)
        self.loss_G_target_GAN_2r = self.get_GAN_loss(self.fake_target_2r, self.discriminator_target)
        # GAN loss of source generator
        self.loss_G_source_GAN_0 = self.get_GAN_loss(self.fake_source_0, self.discriminator_source)
        self.loss_G_source_GAN_1 = self.get_GAN_loss(self.fake_source_1, self.discriminator_source)
        self.loss_G_source_GAN_2f = self.get_GAN_loss(self.fake_source_2f, self.discriminator_source)
        self.loss_G_source_GAN_2r = self.get_GAN_loss(self.fake_source_2r, self.discriminator_source)
        # cycle losses: how far is the reconstruction of fake_2 from real_2?
        self.loss_G_cycle_target = self.criterion_L1(self.fake_fake_target_2, self.real_target_2)
        self.loss_G_cycle_source = self.criterion_L1(self.fake_fake_source_2, self.real_source_2)
        # prediction losses: are both fakes of image 2 the same?
        self.loss_G_cycle_source_time = self.criterion_L1(self.real_source_2, self.fake_source_2r)
        self.loss_G_cycle_target_time = self.criterion_L1(self.real_target_2, self.fake_target_2r)
        # Combine losses and backprop
        self.loss_G += ((self.loss_G_source_GAN_0 + self.loss_G_source_GAN_1 + self.loss_G_source_GAN_2f
                       + self.loss_G_target_GAN_0 + self.loss_G_target_GAN_1 + self.loss_G_target_GAN_2f) / 3
                       + self.cycle_loss_weight * (self.loss_G_cycle_source + self.loss_G_cycle_target
                                                   + self.loss_G_cycle_source_time + self.loss_G_cycle_target_time))
        if self.fix_GAN_loss_2r:
            self.loss_G += self.loss_G_source_GAN_2r + self.loss_G_target_GAN_2r
        super().backward_G()

    def backward_D(self):
        """Calculate discriminator losses"""
        fake_source_0 = self.fake_source_pool.query(self.fake_source_0).detach()
        fake_source_1 = self.fake_source_pool.query(self.fake_source_1).detach()
        fake_source_2f = self.fake_source_pool.query(self.fake_source_2f).detach()
        fake_source_2r = self.fake_source_pool.query(self.fake_source_2r).detach()
        # Target discriminator
        loss_D_target_0, self.loss_D_target_real_0, self.loss_D_target_fake_0 = self.get_GAN_loss_D(
            discriminator=self.discriminator_target,
            real_images=self.real_target_0,
            fake_images=self.fake_target_pool.query(self.fake_target_0),
        )
        loss_D_target_1, self.loss_D_target_real_1, self.loss_D_target_fake_1 = self.get_GAN_loss_D(
            discriminator=self.discriminator_target,
            real_images=self.real_target_1,
            fake_images=self.fake_target_pool.query(self.fake_target_1),
        )
        loss_D_target_2f, self.loss_D_target_real_2, self.loss_D_target_fake_2f = self.get_GAN_loss_D(
            discriminator=self.discriminator_target,
            real_images=self.real_target_2,
            fake_images=self.fake_target_pool.query(self.fake_target_2f),
        )
        loss_D_target_2r, _, self.loss_D_target_fake_2r = self.get_GAN_loss_D(
            discriminator=self.discriminator_target,
            real_images=self.real_target_2,
            fake_images=self.fake_target_pool.query(self.fake_target_2r),
        )
        self.loss_D_target = (loss_D_target_0 + loss_D_target_1 + loss_D_target_2f + loss_D_target_2r) / 4
        # Source discriminator
        loss_D_source_0, self.loss_D_source_real_0, self.loss_D_source_fake_0 = self.get_GAN_loss_D(
            discriminator=self.discriminator_source,
            real_images=self.real_source_0,
            fake_images=self.fake_source_pool.query(self.fake_source_0),
        )
        loss_D_source_1, self.loss_D_source_real_1, self.loss_D_source_fake_1 = self.get_GAN_loss_D(
            discriminator=self.discriminator_source,
            real_images=self.real_source_1,
            fake_images=self.fake_source_pool.query(self.fake_source_1),
        )
        loss_D_source_2f, self.loss_D_source_real_2, self.loss_D_source_fake_2f = self.get_GAN_loss_D(
            discriminator=self.discriminator_source,
            real_images=self.real_source_2,
            fake_images=self.fake_source_pool.query(self.fake_source_2f),
        )
        loss_D_source_2r, _, self.loss_D_source_fake_2r = self.get_GAN_loss_D(
            discriminator=self.discriminator_source,
            real_images=self.real_source_2,
            fake_images=self.fake_source_pool.query(self.fake_source_2r),
        )
        self.loss_D_source = (loss_D_source_0 + loss_D_source_1 + loss_D_source_2f + loss_D_source_2r) / 4
        # Combine losses and backprop
        self.loss_D += self.loss_D_target + self.loss_D_source
        super().backward_D()

    def define_logging(self):
        super().define_logging()
        losses = {
            "Generator/Target/GAN/0": self.loss_G_target_GAN_0,
            "Generator/Target/GAN/1": self.loss_G_target_GAN_1,
            "Generator/Target/GAN/2F": self.loss_G_target_GAN_2f,
            "Generator/Target/GAN/2R": self.loss_G_target_GAN_2r,
            "Generator/Source/GAN/0": self.loss_G_source_GAN_0,
            "Generator/Source/GAN/1": self.loss_G_source_GAN_1,
            "Generator/Source/GAN/2F": self.loss_G_source_GAN_2f,
            "Generator/Source/GAN/2R": self.loss_G_source_GAN_2r,
            "Generator/Cycle/Target": self.loss_G_cycle_target,
            "Generator/Cycle/Source": self.loss_G_cycle_source,
            "Generator/Cycle/Target/Time": self.loss_G_cycle_target_time,
            "Generator/Cycle/Source/Time": self.loss_G_cycle_source_time,
            "Discriminator/Target/Total": self.loss_D_target,
            "Discriminator/Source/Total": self.loss_D_source,
            "Discriminator/Target/Real/0": self.loss_D_target_real_0,
            "Discriminator/Target/Real/1": self.loss_D_target_real_1,
            "Discriminator/Target/Real/2": self.loss_D_target_real_2,
            "Discriminator/Target/Fake/0": self.loss_D_target_fake_0,
            "Discriminator/Target/Fake/1": self.loss_D_target_fake_1,
            "Discriminator/Target/Fake/2F": self.loss_D_target_fake_2f,
            "Discriminator/Target/Fake/2R": self.loss_D_target_fake_2r,
            "Discriminator/Source/Real/0": self.loss_D_source_real_0,
            "Discriminator/Source/Real/1": self.loss_D_source_real_1,
            "Discriminator/Source/Real/2": self.loss_D_source_real_2,
            "Discriminator/Source/Fake/0": self.loss_D_source_fake_0,
            "Discriminator/Source/Fake/1": self.loss_D_source_fake_1,
            "Discriminator/Source/Fake/2F": self.loss_D_source_fake_2f,
            "Discriminator/Source/Fake/2R": self.loss_D_source_fake_2r,
        }
        self.losses = {**self.losses, **losses}

    def test(self):
        with torch.no_grad():
            if self.source_to_target:
                return self.generator_target(self.real_source)
            return self.generator_source(self.real_target)
