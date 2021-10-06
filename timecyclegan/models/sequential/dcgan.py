from timecyclegan.models.base_model import BaseModel
from timecyclegan.models.model_utils import ImagePool
from .sequential_mixin import SequentialMixin


class SequentialDCGAN(SequentialMixin, BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pool_size = 50
        self.generator_architecture = "dcgan"
        self.discriminator_architecture = "dcgan"
        self.generator = self.define_G(
            input_channels=100,
            output_channels=self.target_channels,
        )
        self.encoder = self.define_G(
            input_channels=self.target_channels * self.n_frames_G,
            output_channels=100,
            architecture="dcgan_enc",
        )
        self.generators = {"generator": self.generator, "encoder": self.encoder}
        if self.is_train:
            self.discriminator = self.define_D(self.target_channels)
            self.discriminators = {"discriminator": self.discriminator}
            self.fake_target_pool = ImagePool(self.pool_size)
            self.loss_D_frame, self.loss_D_frame_real, self.loss_D_frame_fake, self.loss_G_GAN_frame = [None] * 4

    def set_input(self, input_data):
        if self.is_train:
            self.real_target = input_data['target'].to(self.device)  # target image of current frame
            self.real_target = self.split_tensor(self.real_target, self.target_channels, self.block_size)

    def _generate_seq(self, input_):
        input_enc = self.encoder(input_)
        return self.generator(input_enc)

    def forward(self):
        self.fake_target = self.forward_sequential(
            generator=self._generate_seq,
            fake_channels=self.target_channels,
        )

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        fake_targets = [self.fake_target_pool.query(fake_target).detach() for fake_target in self.fake_target]
        self.loss_D_frame, self.loss_D_frame_real, self.loss_D_frame_fake = self.get_GAN_loss_D_sequential(
            discriminator=self.discriminator,
            real_images=self.real_target,
            fake_images=fake_targets,
        )
        self.loss_D += self.loss_D_frame
        super().backward_D()

    def backward_G(self):
        self.loss_G_GAN_frame = 0
        for i in range(self.block_size):
            self.loss_G_GAN_frame += self.get_GAN_loss(self.fake_target[i], self.discriminator)
        self.loss_G_GAN_frame /= self.block_size
        self.loss_G += self.loss_G_GAN_frame
        super().backward_G()

    def define_logging(self):
        super().define_logging()
        losses = {
            'Discriminator/Frame': self.loss_D_frame,
            'Discriminator/Frame/Real': self.loss_D_frame_real,
            'Discriminator/Frame/Fake': self.loss_D_frame_fake,
            'Generator/GAN/Frame': self.loss_G_GAN_frame,
        }
        self.losses = {**self.losses, **losses}

    def test(self):
        return self.test_sequential(
            generator=self._generate_seq,
            fake_channels=self.target_channels,
        )
