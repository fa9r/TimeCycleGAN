from timecyclegan.models.base_model import (
    BaseModel,
    PairedModel,
    UnpairedModel
)
from abc import ABC


class SeqDMixinBase(BaseModel, ABC):
    def __init__(self, n_frames_ignored=0, n_frames_D=3, temporal_scales=1, **kwargs):
        super().__init__(**kwargs)
        self.hparam_dict["Sequence Discriminator"] = True
        self.n_frames_ignored = n_frames_ignored
        self.hparam_dict["Ignored Frames in Sequence Losses"] = n_frames_ignored
        self.n_frames_D = n_frames_D
        self.hparam_dict["Sequence Discriminator Input Length"] = self.n_frames_D
        self.temporal_scales = temporal_scales
        self.hparam_dict["Discriminator Temporal Scales"] = temporal_scales

    def define_discriminators_seq(self, input_channels, store_layers=False, disriminator_base_name="discriminator_seq"):
        discriminators_seq = []
        for i in range(self.temporal_scales):
            discriminator_seq = self.define_D(
                input_channels=input_channels,
                store_layers=store_layers,
            )
            discriminators_seq.append(discriminator_seq)
            discriminator_name = disriminator_base_name + (("_" + str(i + 1)) if i > 0 else "")
            self.discriminators[discriminator_name] = discriminator_seq
        return discriminators_seq

    def get_GAN_loss_scaling(self, images, discriminators, is_real=True, conditioned_on=None, store_features_in=None):
        losses = []
        images = images[self.n_frames_ignored:]
        for i, discriminator_seq in enumerate(discriminators):
            loss = 0
            step = self.n_frames_D ** i
            seq_len = (self.n_frames_D - 1) * step
            num_discr = (self.block_size - self.n_frames_ignored - 1) // seq_len if seq_len > 0 else 0
            for j in range(num_discr):
                indices = [j * seq_len + k * step for k in range(self.n_frames_D)]
                loss += self.get_GAN_loss(
                    images=[images[ind] for ind in indices],
                    discriminator=discriminator_seq,
                    is_real=is_real,
                    conditioned_on=conditioned_on,
                    store_features_in=store_features_in[i] if store_features_in is not None else None,
                )
            losses.append((loss / num_discr) if num_discr > 0 else 0)
        return losses

    def get_GAN_loss_scaling_D(self, discriminators, real_images, fake_images, conditioned_on=None):
        fake_images = [fake_image.detach() for fake_image in fake_images]
        losses_real = self.get_GAN_loss_scaling(
            images=real_images,
            discriminators=discriminators,
            is_real=True,
            conditioned_on=conditioned_on,
        )
        losses_fake = self.get_GAN_loss_scaling(
            images=fake_images,
            discriminators=discriminators,
            is_real=False,
            conditioned_on=conditioned_on,
        )
        losses_total = [(loss_real + loss_fake) / 2 for loss_real, loss_fake in zip(losses_real, losses_fake)]
        return losses_total, losses_real, losses_fake


class SeqDMixinBasePaired(SeqDMixinBase, PairedModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.losses_D_seq, self.losses_D_seq_real, self.losses_D_seq_fake = [None] * 3
        self.losses_G_GAN_seq, self.losses_G_fm_seq = [None] * 2

    def get_FM_loss_scaling(
            self,
            fake_images,
            real_images,
            discriminators,
            conditioned_on=None
    ):
        feature_matching_losses = []
        fake_features = [[] for _ in range(self.temporal_scales)]
        real_features = [[] for _ in range(self.temporal_scales)]
        gan_losses = self.get_GAN_loss_scaling(
            images=fake_images,
            discriminators=discriminators,
            conditioned_on=conditioned_on,
            store_features_in=fake_features,
        )
        self.get_GAN_loss_scaling(
            images=real_images,
            discriminators=discriminators,
            conditioned_on=conditioned_on,
            store_features_in=real_features,
        )
        for i in range(self.temporal_scales):
            feature_matching_loss = 0
            for fake_feature, real_feature in zip(fake_features[i], real_features[i]):
                feature_matching_loss += self.criterion_L1(fake_feature, real_feature.detach())
            feature_matching_losses.append(feature_matching_loss)
        return gan_losses, feature_matching_losses

    def define_logging(self):
        super().define_logging()
        losses = {
            **{"Discriminator/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_D_seq)},
            **{"Discriminator/Sequence/%d/Real" % i: loss for i, loss in enumerate(self.losses_D_seq_real)},
            **{"Discriminator/Sequence/%d/Fake" % i: loss for i, loss in enumerate(self.losses_D_seq_fake)},
            **{"Generator/GAN/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_G_GAN_seq)},
            **{"Generator/FeatureMatching/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_G_fm_seq)},
        }
        self.losses = {**self.losses, **losses}


class SeqDMixin(SeqDMixinBase, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discriminators_seq = self.define_discriminators_seq(input_channels=self.target_channels * self.n_frames_D)
        self.losses_D_seq, self.losses_D_seq_real, self.losses_D_seq_fake, self.losses_G_GAN_seq = [None] * 4

    def backward_D(self):
        self.losses_D_seq, self.losses_D_seq_real, self.losses_D_seq_fake = self.get_GAN_loss_scaling_D(
            discriminators=self.discriminators_seq,
            real_images=self.real_target,
            fake_images=self.fake_target,
        )
        self.loss_D += sum(self.losses_D_seq)
        super().backward_D()

    def backward_G(self):
        self.losses_G_GAN_seq = self.get_GAN_loss_scaling(
            images=self.fake_target,
            discriminators=self.discriminators_seq,
        )
        self.loss_G += sum(self.losses_G_GAN_seq)
        super().backward_G()

    def define_logging(self):
        super().define_logging()
        losses = {
            **{"Discriminator/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_D_seq)},
            **{"Discriminator/Sequence/%d/Real" % i: loss for i, loss in enumerate(self.losses_D_seq_real)},
            **{"Discriminator/Sequence/%d/Fake" % i: loss for i, loss in enumerate(self.losses_D_seq_fake)},
            **{"Generator/GAN/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_G_GAN_seq)},
        }
        self.losses = {**self.losses, **losses}


class SeqDMixinPaired(SeqDMixinBasePaired, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discriminators_seq = self.define_discriminators_seq(
            input_channels=self.target_channels * self.n_frames_D,
            store_layers=self.feature_matching_loss_weight > 0,
        )

    def backward_G(self):
        self.losses_G_GAN_seq, self.losses_G_fm_seq = self.get_FM_loss_scaling(
            fake_images=self.fake_target,
            real_images=self.real_target,
            discriminators=self.discriminators_seq,
        )
        self.loss_G += sum(self.losses_G_GAN_seq)
        self.loss_G += self.feature_matching_loss_weight * sum(self.losses_G_fm_seq)
        super().backward_G()

    def backward_D(self):
        self.losses_D_seq, self.losses_D_seq_real, self.losses_D_seq_fake = self.get_GAN_loss_scaling_D(
            discriminators=self.discriminators_seq,
            real_images=self.real_target,
            fake_images=self.fake_target,
        )
        self.loss_D += sum(self.losses_D_seq)
        super().backward_D()


class CondSeqDMixinPaired(SeqDMixinBasePaired, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        total_channels = self.source_channels + self.target_channels
        self.discriminators_seq = self.define_discriminators_seq(
            input_channels=total_channels * self.n_frames_D,
            store_layers=self.feature_matching_loss_weight > 0,
        )

    def backward_G(self):
        self.losses_G_GAN_seq, self.losses_G_fm_seq = self.get_FM_loss_scaling(
            fake_images=self.fake_target,
            real_images=self.real_target,
            discriminators=self.discriminators_seq,
            conditioned_on=self.real_source,
        )
        self.loss_G += sum(self.losses_G_GAN_seq)
        self.loss_G += self.feature_matching_loss_weight * sum(self.losses_G_fm_seq)
        super().backward_G()

    def backward_D(self):
        self.losses_D_seq, self.losses_D_seq_real, self.losses_D_seq_fake = self.get_GAN_loss_scaling_D(
            discriminators=self.discriminators_seq,
            real_images=self.real_target,
            fake_images=self.fake_target,
            conditioned_on=self.real_source,
        )
        self.loss_D += sum(self.losses_D_seq)
        super().backward_D()


class SeqDMixinUnpaired(SeqDMixinBase, UnpairedModel, ABC):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.discriminators_target_seq = self.define_discriminators_seq(
            input_channels=self.target_channels * self.n_frames_D,
            disriminator_base_name="discriminator_target_seq",
        )
        self.discriminators_source_seq = self.define_discriminators_seq(
            input_channels=self.source_channels * self.n_frames_D,
            disriminator_base_name="discriminator_source_seq",
        )
        self.losses_G_target_GAN_seq, self.losses_G_source_GAN_seq = [None] * 2
        self.losses_D_source_seq, self.losses_D_target_seq = [None] * 2
        self.losses_D_source_seq_real, self.losses_D_source_seq_fake = [None] * 2
        self.losses_D_target_seq_real, self.losses_D_target_seq_fake = [None] * 2

    def backward_G(self):
        # source
        self.losses_G_source_GAN_seq = self.get_GAN_loss_scaling(
            images=self.fake_source,
            discriminators=self.discriminators_source_seq,
        )
        # target
        self.losses_G_target_GAN_seq = self.get_GAN_loss_scaling(
            images=self.fake_target,
            discriminators=self.discriminators_target_seq,
        )
        self.loss_G += sum(self.losses_G_source_GAN_seq) + sum(self.losses_G_target_GAN_seq)
        super().backward_G()

    def backward_D(self):
        # source
        self.losses_D_source_seq, self.losses_D_source_seq_real, self.losses_D_source_seq_fake = self.get_GAN_loss_scaling_D(
            discriminators=self.discriminators_source_seq,
            real_images=self.real_source,
            fake_images=self.fake_source,
        )
        # target
        self.losses_D_target_seq, self.losses_D_target_seq_real, self.losses_D_target_seq_fake = self.get_GAN_loss_scaling_D(
            discriminators=self.discriminators_target_seq,
            real_images=self.real_target,
            fake_images=self.fake_target,
        )
        self.loss_D += sum(self.losses_D_source_seq) + sum(self.losses_D_target_seq)
        super().backward_D()

    def define_logging(self):
        super().define_logging()
        losses = {
            **{"Discriminator/Target/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_D_target_seq)},
            **{"Discriminator/Target/Sequence/%d/Real" % i: loss
               for i, loss in enumerate(self.losses_D_target_seq_real)},
            **{"Discriminator/Target/Sequence/%d/Fake" % i: loss
               for i, loss in enumerate(self.losses_D_target_seq_fake)},
            **{"Discriminator/Source/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_D_source_seq)},
            **{"Discriminator/Source/Sequence/%d/Real" % i: loss
               for i, loss in enumerate(self.losses_D_source_seq_real)},
            **{"Discriminator/Source/Sequence/%d/Fake" % i: loss
               for i, loss in enumerate(self.losses_D_source_seq_fake)},
            **{"Generator/Target/GAN/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_G_target_GAN_seq)},
            **{"Generator/Source/GAN/Sequence/%d" % i: loss for i, loss in enumerate(self.losses_G_source_GAN_seq)},
        }
        self.losses = {**self.losses, **losses}
