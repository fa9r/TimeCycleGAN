from timecyclegan.models.base_model import PairedModel
from .sequential_mixin import SequentialMixin


class SequentialPix2Pix(SequentialMixin, PairedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.generator = self.define_G(
            self.source_channels * (self.n_frames_G_condition + 1) + self.target_channels * self.n_frames_G,
            self.target_channels
        )
        self.generators = {"generator": self.generator}
        if self.is_train:
            self.discriminator = self.define_D(
                input_channels=self.source_channels + self.target_channels,
                store_layers=self.feature_matching_loss_weight > 0,
            )
            self.discriminators = {"discriminator": self.discriminator}
            # initialize losses; these will be set during the backward pass when calling optimize_parameters()
            self.loss_D_frame, self.loss_D_frame_real, self.loss_D_frame_fake = [None] * 3
            self.loss_G_GAN_frame, self.loss_G_fm_frame,  self.loss_G_L1, self.loss_G_perceptual = [None] * 4

    def set_input(self, input_data):
        self.real_source = input_data['source'].to(self.device)  # source image of current frame
        if self.is_train:
            self.real_target = input_data['target'].to(self.device)  # target image of current frame
            # split into list of individual frames
            self.real_target = self.split_tensor(self.real_target, self.target_channels, self.block_size)
            self.real_source = self.split_tensor(self.real_source, self.source_channels, self.block_size)

    def forward(self):
        self.fake_target = self.forward_sequential(
            generator=self.generator,
            fake_channels=self.target_channels,
            cond=self.real_source,
            cond_channels=self.source_channels,
        )

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        self.loss_D_frame, self.loss_D_frame_real, self.loss_D_frame_fake = self.get_GAN_loss_D_sequential(
            discriminator=self.discriminator,
            real_images=self.real_target,
            fake_images=self.fake_target,
            conditioned_on=self.real_source,
        )
        self.loss_D += self.loss_D_frame
        super().backward_D()

    def backward_G(self):
        loss_gan, loss_fm, loss_l1, loss_perceptual = [0] * 4
        for i in range(self.block_size):
            real_source, real_target, fake_target = self.real_source[i], self.real_target[i], self.fake_target[i]
            gan_loss, fm_loss = self.get_FM_loss(
                fake_images=fake_target,
                real_images=real_target,
                discriminator=self.discriminator,
                conditioned_on=real_source,
            )
            loss_gan += gan_loss  # GAN loss
            loss_fm += fm_loss  # FM loss
            loss_l1 += self.criterion_L1(fake_target, real_target)  # L1 losses
            loss_perceptual += self.criterion_VGG(fake_target, real_target)  # VGG loss
        self.loss_G_GAN_frame, self.loss_G_fm_frame, self.loss_G_L1, self.loss_G_perceptual = [
            loss / self.block_size for loss in [loss_gan, loss_fm, loss_l1, loss_perceptual]
        ]
        self.loss_G += self.loss_G_GAN_frame
        self.loss_G += self.feature_matching_loss_weight * self.loss_G_fm_frame
        self.loss_G += self.l1_loss_weight * self.loss_G_L1
        self.loss_G += self.perceptual_loss_weight * self.loss_G_perceptual
        super().backward_G()

    def define_logging(self):
        super().define_logging()
        losses = {
            'Discriminator/Frame': self.loss_D_frame,
            'Discriminator/Frame/Real': self.loss_D_frame_real,
            'Discriminator/Frame/Fake': self.loss_D_frame_fake,
            'Generator/L1': self.loss_G_L1,
            'Generator/GAN/Frame': self.loss_G_GAN_frame,
            'Generator/FeatureMatching/Frame': self.loss_G_fm_frame,
            'Generator/Perceptual': self.loss_G_perceptual,
        }
        self.losses = {**self.losses, **losses}

    def test(self):
        return self.test_sequential(
            generator=self.generator,
            fake_channels=self.target_channels,
            cond=self.real_source,
            cond_channels=self.source_channels,
        )
