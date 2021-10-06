"""Reimplemented CycleGAN (https://junyanz.github.io/CycleGAN/) model"""

import torch

from timecyclegan.models.base_model import UnpairedModel
from timecyclegan.models.model_utils import ImagePool


class CycleGANModel(UnpairedModel):
    """CycleGAN Model"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.identity_loss_weight = 0.5 * self.cycle_loss_weight
        self.pool_size = 50
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
            # initialize image buffers to store previously generated images for discriminator training
            self.fake_source_pool, self.fake_target_pool = ImagePool(self.pool_size), ImagePool(self.pool_size)
            # Define other generations that will be set in forward() during training
            self.fake_fake_source, self.fake_fake_target, self.fake_idt_source, self.fake_idt_target = [None] * 4
            # Define losses that will be set in optimize_parameters() during training
            self.loss_G_target_GAN, self.loss_G_target_idt, self.loss_G_source_GAN, self.loss_G_source_idt = [None] * 4
            self.loss_D_target, self.loss_D_target_real, self.loss_D_target_fake = [None] * 3
            self.loss_D_source, self.loss_D_source_real, self.loss_D_source_fake = [None] * 3
            self.loss_G_cycle_source, self.loss_G_cycle_target = [None] * 2

    def set_input(self, input_data):
        if self.is_train:
            source, target = input_data
            self.real_source = source["source"].to(self.device)
            self.real_target = target["target"].to(self.device)
        elif self.source_to_target:
            self.real_source = input_data["source"].to(self.device)
        else:
            self.real_target = input_data["target"].to(self.device)

    def forward(self):
        self.fake_target = self.generator_target(self.real_source)
        self.fake_source = self.generator_source(self.real_target)
        self.fake_fake_source = self.generator_source(self.fake_target)
        self.fake_fake_target = self.generator_target(self.fake_source)
        self.fake_idt_source = self.generator_source(self.real_source)
        self.fake_idt_target = self.generator_target(self.real_target)

    def backward_G(self):
        # GAN loss of target generator
        self.loss_G_target_GAN = self.get_GAN_loss(self.fake_target, self.discriminator_target)
        # GAN loss of source generator
        self.loss_G_source_GAN = self.get_GAN_loss(self.fake_source, self.discriminator_source)
        # Cycle losses
        self.loss_G_cycle_source = self.criterion_L1(self.real_source, self.fake_fake_source)
        self.loss_G_cycle_target = self.criterion_L1(self.real_target, self.fake_fake_target)
        # Identity losses
        self.loss_G_source_idt = self.criterion_L1(self.real_source, self.fake_idt_source)
        self.loss_G_target_idt = self.criterion_L1(self.real_target, self.fake_idt_target)
        # Combine losses and backprop
        self.loss_G += self.loss_G_target_GAN + self.loss_G_source_GAN
        self.loss_G += self.cycle_loss_weight * (self.loss_G_cycle_source + self.loss_G_cycle_target)
        self.loss_G += self.identity_loss_weight * (self.loss_G_source_idt + self.loss_G_target_idt)
        super().backward_G()

    def backward_D(self):
        # Target discriminator
        self.loss_D_target, self.loss_D_target_real, self.loss_D_target_fake = self.get_GAN_loss_D(
            discriminator=self.discriminator_target,
            real_images=self.real_target,
            fake_images=self.fake_target_pool.query(self.fake_target),
        )
        # Source discriminator
        self.loss_D_source, self.loss_D_source_real, self.loss_D_source_fake = self.get_GAN_loss_D(
            discriminator=self.discriminator_source,
            real_images=self.real_source,
            fake_images=self.fake_source_pool.query(self.fake_source),
        )
        # Combine losses and backprop
        self.loss_D += self.loss_D_target + self.loss_D_source
        super().backward_D()

    def define_logging(self):
        super().define_logging()
        losses = {
            "Generator/Target/GAN": self.loss_G_target_GAN,
            "Generator/Source/GAN": self.loss_G_source_GAN,
            "Discriminator/Target/Total": self.loss_D_target,
            "Discriminator/Target/Real": self.loss_D_target_real,
            "Discriminator/Target/Fake": self.loss_D_target_fake,
            "Discriminator/Source/Total": self.loss_D_source,
            "Discriminator/Source/Real": self.loss_D_source_real,
            "Discriminator/Source/Fake": self.loss_D_source_fake,
            "Generator/Cycle/Source": self.loss_G_cycle_source,
            "Generator/Cycle/Target": self.loss_G_cycle_target,
            "Generator/Source/Identity": self.loss_G_source_idt,
            "Generator/Target/Identity": self.loss_G_target_idt,
        }
        self.losses = {**self.losses, **losses}

    def test(self):
        with torch.no_grad():
            if self.source_to_target:
                return self.generator_target(self.real_source)
            return self.generator_source(self.real_target)
