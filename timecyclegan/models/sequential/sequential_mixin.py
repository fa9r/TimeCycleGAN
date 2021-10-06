# pylint: disable=abstract-method

import torch

from timecyclegan.models.base_model import BaseModel


class SequentialMixin(BaseModel):
    def __init__(self, n_frames_G=1, G_input_conditioning=False, **kwargs):
        """
       Initialize the model
       :param n_frames_G: How many previous generations to condition G on
       :param G_input_conditioning: Whether to also condition G on previous sources
       :param kwargs: Additional keyword arguments for BaseModel
       """
        super().__init__(**kwargs)
        self.n_frames_G = n_frames_G
        self.hparam_dict["G Prev Frames"] = n_frames_G
        self.n_frames_G_condition = n_frames_G if G_input_conditioning else 0
        self.hparam_dict["G Prev Conditioning"] = self.n_frames_G_condition
        self.first_test_input = True
        self.fake_prev, self.conditioning_prev = [None] * 2

    def forward_sequential(self, generator, fake_channels, cond=None, cond_channels=0, seq_len=None):
        prev_fakes = self.get_placeholder_images(self.n_frames_G, channels=fake_channels)
        conditioning = self.get_placeholder_images(self.n_frames_G_condition, channels=cond_channels)
        seq_len = seq_len if seq_len is not None else self.block_size
        for i in range(seq_len):
            if cond is not None:
                conditioning.append(cond[i])
                fake_target = generator(torch.cat((*conditioning[i:], *prev_fakes[i:]), 1))
            else:
                fake_target = generator(torch.cat(prev_fakes[i:], 1))
            prev_fakes.append(fake_target)
        return prev_fakes[-seq_len:]

    def get_GAN_loss_D_sequential(self, discriminator, real_images, fake_images, conditioned_on=None, seq_len=None):
        loss_real, loss_fake = [0] * 2
        seq_len = seq_len if seq_len is not None else self.block_size
        for i in range(seq_len):
            conditioning = conditioned_on[i] if conditioned_on is not None else None
            loss_real += self.get_GAN_loss(real_images[i], discriminator, True, conditioning)
            loss_fake += self.get_GAN_loss(fake_images[i].detach(), discriminator, False, conditioning)
        loss_total = (loss_real + loss_fake) / 2
        return [loss / seq_len for loss in [loss_total, loss_real, loss_fake]] if seq_len > 0 else [0] * 3

    def test_sequential(self, generator, fake_channels, cond=None, cond_channels=0):
        with torch.no_grad():
            # init prev generations/sources with random noise
            if self.first_test_input:
                self.fake_prev = self.get_placeholder_images(self.n_frames_G, channels=fake_channels)
                self.conditioning_prev = self.get_placeholder_images(self.n_frames_G_condition, channels=cond_channels)
                self.first_test_input = False
            if cond is not None:
                self.conditioning_prev.append(cond)  # save previous source
                fake = generator(torch.cat((*self.conditioning_prev, *self.fake_prev), 1))
            else:
                fake = generator(torch.cat(self.fake_prev, 1))
            self.fake_prev.append(fake)  # save previous generation
            if cond is not None:
                del self.conditioning_prev[0]  # pop oldest sources
            del self.fake_prev[0]  # pop oldest generation
        return fake
