"""
DCGAN model
Adjusted from official PyTorch tutorial https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
"""

from timecyclegan.models.base_model import BaseModel


class DCGANModel(BaseModel):
    """DCGAN model"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator = self.define_G(100, self.target_channels, 'dcgan')
        self.generators = {"generator": self.generator}
        if self.is_train:
            self.discriminator = self.define_D(self.target_channels, 'dcgan')
            self.discriminators = {"discriminator": self.discriminator}
            self.loss_D_real, self.loss_D_fake = [None] * 2

    def set_input(self, input_data):
        if self.is_train:
            self.real_target = input_data['target'].to(self.device)

    def forward(self):
        noise = self.get_placeholder_image(channels=100, width=1, height=1, type="normal")
        self.fake_target = self.generator(noise)

    def backward_G(self):
        self.loss_G += self.get_GAN_loss(self.fake_target, self.discriminator)
        super().backward_G()

    def backward_D(self):
        loss_D, self.loss_D_real, self.loss_D_fake = self.get_GAN_loss_D(
            discriminator=self.discriminator,
            real_images=self.real_target,
            fake_images=self.fake_target
        )
        self.loss_D += loss_D
        super().backward_D()

    def define_logging(self):
        super().define_logging()
        losses = {
            'Generator/Total': self.loss_G,
            'Discriminator/Total': self.loss_D,
            'Discriminator/Real': self.loss_D_real,
            'Discriminator/Fake': self.loss_D_fake,
        }
        self.losses = {**self.losses, **losses}

    def test(self):
        return super().test()
