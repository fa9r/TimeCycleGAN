from timecyclegan.models.mixins import TimeCycleMixin, SeqDMixin
from .sequential import SequentialDCGAN


class TimeCycleDCGAN(TimeCycleMixin, SeqDMixin, SequentialDCGAN):
    def __init__(self, timecycle_type="unconditional", **kwargs):
        assert timecycle_type == "unconditional"
        super().__init__(timecycle_type=timecycle_type, **kwargs)
        self.define_timecycle(
            input_channels=self.target_channels * 2,
            output_channels=self.target_channels,
        )
        self.add_timecycle_discriminator(self.discriminator, 1)

    def forward(self):
        super().forward()
        self.timecycle_forward(sequence=self.fake_target)
        self.set_timecycle_discriminator_input(real_sequence=self.real_target)
