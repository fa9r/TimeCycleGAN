from timecyclegan.models.base_model import UnpairedModel
from timecyclegan.models.model_utils import ImagePool
from .sequential_mixin import SequentialMixin


class SequentialCycleGAN(SequentialMixin, UnpairedModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.identity_loss_weight = 0.5 * self.cycle_loss_weight
        self.pool_size = 0
        # Define generators
        self.generator_target = self.define_G(
            self.source_channels * (self.n_frames_G_condition + 1) + self.target_channels * self.n_frames_G,
            self.target_channels
        )
        self.generator_source = self.define_G(
            self.target_channels * (self.n_frames_G_condition + 1) + self.source_channels * self.n_frames_G,
            self.source_channels
        )
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
            self.real_source = self.split_tensor(self.real_source, self.source_channels, self.block_size)
            self.real_target = target["target"].to(self.device)
            self.real_target = self.split_tensor(self.real_target, self.target_channels, self.block_size)
        elif self.source_to_target:
            self.real_source = input_data["source"].to(self.device)
        else:
            self.real_target = input_data["target"].to(self.device)

    def forward(self):
        def forward_sequential_source(cond, cond_channels):
            return self.forward_sequential(
                generator=self.generator_source,
                fake_channels=self.source_channels,
                cond=cond,
                cond_channels=cond_channels,
            )

        def forward_sequential_target(cond, cond_channels):
            return self.forward_sequential(
                generator=self.generator_target,
                fake_channels=self.target_channels,
                cond=cond,
                cond_channels=cond_channels,
            )

        # generate fake source/target
        self.fake_target = forward_sequential_target(
            cond=self.real_source,
            cond_channels=self.source_channels,
        )
        self.fake_source = forward_sequential_source(
            cond=self.real_target,
            cond_channels=self.target_channels,
        )
        # generate reconstructions for source/target
        self.fake_fake_target = forward_sequential_target(
            cond=self.fake_source,
            cond_channels=self.source_channels,
        )
        self.fake_fake_source = forward_sequential_source(
            cond=self.fake_target,
            cond_channels=self.target_channels,
        )
        # generate identity loss fake source/target
        self.fake_idt_target = forward_sequential_target(
            cond=self.real_target,
            cond_channels=self.target_channels,
        )
        self.fake_idt_source = forward_sequential_source(
            cond=self.real_source,
            cond_channels=self.source_channels,
        )

    def backward_D(self):
        # Target discriminator
        fake_targets = [self.fake_target_pool.query(fake_target).detach() for fake_target in self.fake_target]
        self.loss_D_target, self.loss_D_target_real, self.loss_D_target_fake = self.get_GAN_loss_D_sequential(
            discriminator=self.discriminator_target,
            real_images=self.real_target,
            fake_images=fake_targets,
        )
        # Source discriminator
        fake_sources = [self.fake_source_pool.query(fake_source).detach() for fake_source in self.fake_source]
        self.loss_D_source, self.loss_D_source_real, self.loss_D_source_fake = self.get_GAN_loss_D_sequential(
            discriminator=self.discriminator_source,
            real_images=self.real_source,
            fake_images=fake_sources,
        )
        # Combine losses and backprop
        self.loss_D += self.loss_D_target + self.loss_D_source
        super().backward_D()

    def backward_G(self):
        loss_gan_target, loss_gan_source, loss_cycle_target, loss_cycle_source, loss_idt_target, loss_idt_source = [0]*6
        for i in range(self.block_size):
            # GAN loss of target generator
            loss_gan_target += self.get_GAN_loss(self.fake_target[i], self.discriminator_target)
            # GAN loss of source generator
            loss_gan_source += self.get_GAN_loss(self.fake_source[i], self.discriminator_source)
            # Cycle losses
            loss_cycle_source += self.criterion_L1(self.real_source[i], self.fake_fake_source[i])
            loss_cycle_target += self.criterion_L1(self.real_target[i], self.fake_fake_target[i])
            # Identity losses
            loss_idt_source += self.criterion_L1(self.real_source[i], self.fake_idt_source[i])
            loss_idt_target += self.criterion_L1(self.real_target[i], self.fake_idt_target[i])
        # divide all losses by block_size
        (self.loss_G_target_GAN, self.loss_G_source_GAN, self.loss_G_cycle_target, self.loss_G_cycle_source,
         self.loss_G_target_idt, self.loss_G_source_idt) = [
            loss / self.block_size for loss in [
                loss_gan_target, loss_gan_source, loss_cycle_target, loss_cycle_source, loss_idt_target, loss_idt_source
            ]
        ]
        # Combine losses and backprop
        self.loss_G += self.loss_G_target_GAN + self.loss_G_source_GAN
        self.loss_G += self.cycle_loss_weight * (self.loss_G_cycle_source + self.loss_G_cycle_target)
        self.loss_G += self.identity_loss_weight * (self.loss_G_source_idt + self.loss_G_target_idt)
        super().backward_G()

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
        images = {
            "Generations/Cycle/Source": self.fake_fake_source,
            "Generations/Cycle/Target": self.fake_fake_target,
            "Generations/Identity/Source": self.fake_idt_source,
            "Generations/Identity/Target": self.fake_idt_target,
        }
        self.images = {**self.images, **images}

    def test(self):
        if self.source_to_target:
            return self.test_sequential(
                generator=self.generator_target,
                cond=self.real_source,
                fake_channels=self.target_channels,
                cond_channels=self.source_channels,
            )
        return self.test_sequential(
            generator=self.generator_source,
            cond=self.real_target,
            fake_channels=self.source_channels,
            cond_channels=self.target_channels
        )
