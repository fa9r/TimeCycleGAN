"""Modified Pix2Pix model that can predict videos block-wise (multiple subsequent images at a time)"""

from timecyclegan.models.base_model import PairedModel


class Pix2PixModel(PairedModel):
    """
    Generalized Pix2Pix model that can process videos block-wise.
    Identical to Pix2Pix if block_size is set to 1.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # define generator
        self.generator = self.define_G(self.source_channels * self.block_size, self.target_channels * self.block_size)
        self.generators = {"generator": self.generator}
        if self.is_train:
            # define discriminator
            self.discriminator = self.define_D((self.source_channels + self.target_channels) * self.block_size)
            self.discriminators = {"discriminator": self.discriminator}
            # initialize losses; these will be set during the backward pass when calling optimize_parameters().
            self.loss_G_GAN, self.loss_G_L1, self.loss_D_real, self.loss_D_fake = [None] * 4

    def set_input(self, input_data):
        self.real_source = input_data['source'].to(self.device)
        if self.is_train:
            self.real_target = input_data['target'].to(self.device)

    def forward(self):
        self.fake_target = self.generator(self.real_source)

    def backward_D(self):
        loss_D, self.loss_D_real, self.loss_D_fake = self.get_GAN_loss_D(
            discriminator=self.discriminator,
            real_images=self.real_target,
            fake_images=self.fake_target,
            conditioned_on=self.real_source,
        )
        self.loss_D += loss_D
        super().backward_D()

    def backward_G(self):
        # GAN loss
        self.loss_G_GAN = self.get_GAN_loss(self.fake_target, self.discriminator, conditioned_on=self.real_source)
        # L1 loss
        self.loss_G_L1 = self.criterion_L1(self.fake_target, self.real_target) * self.l1_loss_weight
        # combine losses and calculate gradients
        self.loss_G += self.loss_G_GAN + self.loss_G_L1
        super().backward_G()

    def define_logging(self):
        super().define_logging()
        losses = {
            'Generator/GAN': self.loss_G_GAN,
            'Generator/L1': self.loss_G_L1,
            'Discriminator/Real': self.loss_D_real,
            'Discriminator/Fake': self.loss_D_fake,
        }
        self.losses = {**self.losses, **losses}

    def test(self):
        # If self.block_size > 1, we only return predictions for the image in the middle (at block_size//2)
        middle_block_index = (self.block_size // 2) * self.target_channels
        return super().test()[:, middle_block_index:middle_block_index + self.target_channels, :, :]
