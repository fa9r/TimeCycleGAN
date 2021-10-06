import torch

from .cyclegan import SequentialCycleGAN


class SequentialRecycleGAN(SequentialCycleGAN):
    def __init__(
            self,
            recycle_loss_weight=0,
            recycle_predictor_architecture="resnet_6blocks",
            recycle_predictor_filters=64,
            **kwargs
    ):
        super().__init__(**kwargs)
        if self.is_train:
            # Motion Predictors
            self.recycle_loss_weight = recycle_loss_weight
            self.hparam_dict["Recycle Loss Weight"] = recycle_loss_weight
            self.hparam_dict["Recycle Predictor Architecture"] = recycle_predictor_architecture
            self.hparam_dict["Recycle Predictor Filters"] = recycle_predictor_filters
            self.predictor_target = self.define_G(
                input_channels=self.target_channels * 2,
                output_channels=self.target_channels,
                architecture=recycle_predictor_architecture,
                filters=recycle_predictor_filters,
            )
            self.predictor_source = self.define_G(
                input_channels=self.source_channels * 2,
                output_channels=self.source_channels,
                architecture=recycle_predictor_architecture,
                filters=recycle_predictor_filters,
            )
            self.generators["predictor_target"] = self.predictor_target
            self.generators["predictor_source"] = self.predictor_source
            # Define additional fakes that will be set in forward()
            self.pred_target_fake, self.pred_target_real, self.pred_source_fake, self.pred_source_real = [None] * 4
            self.fake_target_recycle, self.fake_source_recycle = [None] * 2
            # Define additional losses that will be set in optimize_parameters() during training
            self.loss_D_preds = None
            self.loss_D_target_pred_fake, self.loss_D_target_pred_real = [None] * 2
            self.loss_D_source_pred_fake, self.loss_D_source_pred_real = [None] * 2
            self.loss_G_target_GAN_pred_fake, self.loss_G_target_GAN_pred_real = [None] * 2
            self.loss_G_source_GAN_pred_fake, self.loss_G_source_GAN_pred_real = [None] * 2
            self.loss_G_recycle_target, self.loss_G_recycle_source = [None] * 2
            self.loss_G_pred_target, self.loss_G_pred_source = [None] * 2

    def forward(self):
        super().forward()
        self.pred_target_fake, self.pred_target_real, self.pred_source_fake, self.pred_source_real = [], [], [], []
        self.fake_source_recycle, self.fake_target_recycle = [], []
        for i in range(self.block_size - 2):
            # generate predictions on fake images
            self.pred_target_fake.append(self.predictor_target(torch.cat(self.fake_target[i:i+2], 1)))
            self.pred_source_fake.append(self.predictor_source(torch.cat(self.fake_source[i:i+2], 1)))
            # generate predictor predictions on real images
            self.pred_target_real.append(self.predictor_target(torch.cat(self.real_target[i:i+2], 1)))
            self.pred_source_real.append(self.predictor_source(torch.cat(self.real_source[i:i+2], 1)))
        # generate recycle reconstructions
        self.fake_target_recycle = self.forward_sequential(
            generator=self.generator_target,
            fake_channels=self.target_channels,
            cond=self.pred_source_fake,
            cond_channels=self.source_channels,
            seq_len=self.block_size - 2,
        )
        self.fake_source_recycle = self.forward_sequential(
            generator=self.generator_source,
            fake_channels=self.source_channels,
            cond=self.pred_target_fake,
            cond_channels=self.target_channels,
            seq_len=self.block_size - 2,
        )

    def backward_D(self):

        def get_GAN_loss_source(fake_images, is_source=True):
            return self.get_GAN_loss_D_sequential(
                discriminator=self.discriminator_source if is_source else self.discriminator_target,
                real_images=self.real_source[2:] if is_source else self.real_target[2:],
                fake_images=fake_images,
                seq_len=self.block_size - 2,
            )

        def get_GAN_loss_target(fake_images):
            return get_GAN_loss_source(fake_images, is_source=False)

        self.loss_D_preds = 0
        total_loss, _, self.loss_D_target_pred_fake = get_GAN_loss_target(self.pred_target_fake)
        self.loss_D_preds += total_loss
        total_loss, _, self.loss_D_target_pred_real = get_GAN_loss_target(self.pred_target_real)
        self.loss_D_preds += total_loss
        total_loss, _, self.loss_D_source_pred_fake = get_GAN_loss_source(self.pred_source_fake)
        self.loss_D_preds += total_loss
        total_loss, _, self.loss_D_source_pred_real = get_GAN_loss_source(self.pred_source_real)
        self.loss_D_preds += total_loss
        self.loss_D += self.loss_D_preds
        super().backward_D()

    def backward_G(self):
        loss_gan_target_pred_fake, loss_gan_target_pred_real = [0] * 2
        loss_gan_source_pred_fake, loss_gan_source_pred_real = [0] * 2
        loss_recycle_target, loss_recycle_source, loss_motion_target, loss_motion_source = [0] * 4
        for i in range(self.block_size - 2):
            # GAN losses
            loss_gan_target_pred_fake += self.get_GAN_loss(self.pred_target_fake[i], self.discriminator_target)
            loss_gan_target_pred_real += self.get_GAN_loss(self.pred_target_real[i], self.discriminator_target)
            loss_gan_source_pred_fake += self.get_GAN_loss(self.pred_source_fake[i], self.discriminator_target)
            loss_gan_source_pred_real += self.get_GAN_loss(self.pred_source_real[i], self.discriminator_target)
            # Recycle losses
            loss_recycle_target += self.criterion_L1(self.real_target[i+2], self.fake_target_recycle[i])
            loss_recycle_source += self.criterion_L1(self.real_source[i+2], self.fake_source_recycle[i])
            # Motion losses
            loss_motion_target += self.criterion_L1(self.real_target[i+2], self.pred_target_real[i])
            loss_motion_source += self.criterion_L1(self.real_source[i+2], self.pred_source_real[i])

        # divide all losses by block_size
        def divide_loss(loss):
            return loss / (self.block_size - 2) if self.block_size > 2 else 0

        self.loss_G_target_GAN_pred_fake = divide_loss(loss_gan_target_pred_fake)
        self.loss_G_target_GAN_pred_real = divide_loss(loss_gan_target_pred_real)
        self.loss_G_source_GAN_pred_fake = divide_loss(loss_gan_source_pred_fake)
        self.loss_G_source_GAN_pred_real = divide_loss(loss_gan_source_pred_real)
        self.loss_G_recycle_target = divide_loss(loss_recycle_target)
        self.loss_G_recycle_source = divide_loss(loss_recycle_source)
        self.loss_G_pred_target = divide_loss(loss_motion_target)
        self.loss_G_pred_source = divide_loss(loss_motion_source)
        # Combine losses and backprop
        self.loss_G += self.loss_G_target_GAN_pred_fake + self.loss_G_target_GAN_pred_real
        self.loss_G += self.loss_G_source_GAN_pred_fake + self.loss_G_source_GAN_pred_real
        self.loss_G += self.recycle_loss_weight * (self.loss_G_recycle_target + self.loss_G_recycle_source)
        self.loss_G += self.recycle_loss_weight * (self.loss_G_pred_target + self.loss_G_pred_source)
        super().backward_G()

    def define_logging(self):
        super().define_logging()
        losses = {
            "Discriminator/Preds": self.loss_D_preds,
            "Discriminator/Preds/Target/Fake": self.loss_D_target_pred_fake,
            "Discriminator/Preds/Target/Real": self.loss_D_target_pred_real,
            "Discriminator/Preds/Source/Fake": self.loss_D_source_pred_fake,
            "Discriminator/Preds/Source/Real": self.loss_D_source_pred_real,
            "Generator/Target/GAN/Pred/Fake": self.loss_G_target_GAN_pred_fake,
            "Generator/Target/GAN/Pred/Real": self.loss_G_target_GAN_pred_real,
            "Generator/Source/GAN/Pred/Fake": self.loss_G_source_GAN_pred_fake,
            "Generator/Source/GAN/Pred/Real": self.loss_G_source_GAN_pred_real,
            "Generator/Recycle/Target": self.loss_G_recycle_target,
            "Generator/Recycle/Source": self.loss_G_recycle_source,
            "Generator/Predictor/Target": self.loss_G_pred_target,
            "Generator/Predictor/Source": self.loss_G_pred_source,
        }
        self.losses = {**self.losses, **losses}
        images = {
            "Generations/Recycle/Target": self.fake_target_recycle,
            "Generations/Recycle/Source": self.fake_source_recycle,
            "Generations/Predictor/Target": self.pred_target_real,
            "Generations/Predictor/Source": self.pred_source_real,
        } if self.block_size > 2 else {}
        self.images = {**self.images, **images}
